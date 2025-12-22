#!/usr/bin/env python3
# Produces various CMB map realizations.

from __future__ import division, print_function

import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from orphics import maps, io, stats, cosmology, lensing
from pixell import enmap as penmap
from pixell.fft import ifft, fft
import camb

def search_dicts_key(ds_to_search, search_key):
    for ds in ds_to_search:
        if search_key in ds:
            return ds
    return None

class SimSetupCMB:
    """
    A class used as a container for information needed for producing CMB
    simulations
    """
    def __init__(self, cambRootfile=None, m_width_deg=10, pix_res_arcmin=2.34375, taperwdeg=1.5, theory_orph=None):
        # geometry of map
        self.shape, self.wcs = maps.rect_geometry(width_deg=m_width_deg, px_res_arcmin=pix_res_arcmin)
        self.shape = (3,)+ self.shape
        self.modlmap = penmap.modlmap(self.shape, self.wcs)
        self.fc = maps.FourierCalc(self.shape, self.wcs)
        # theory spectra for map
        if cambRootfile is not None:
            self.theory = cosmology.loadTheorySpectraFromCAMB(
                cambRootfile, unlensedEqualsLensed=False, useTotal=False, TCMB=2.7255e6, lpad=9000, get_dimensionless=False)
        else:
            self.theory= theory_orph
        self.cltautau = None
        self.incl_tau = False
        self.clalphalpha = None
        self.incl_cbfringe = False
        self.varied_cosmo_params = None
        # map mask (includes polarization mask)
        self.tellminmax = [300, 3500]
        self.pellminmax = [300, 3500]
        self.kellminmax = [20, 3500]
        self.tmask = maps.mask_kspace(self.shape, self.wcs, lmin=self.tellminmax[0], lmax=self.tellminmax[1])
        self.pmask = maps.mask_kspace(self.shape, self.wcs, lmin=self.pellminmax[0], lmax=self.pellminmax[1])
        self.kmask = maps.mask_kspace(self.shape, self.wcs, lmin=self.kellminmax[0], lmax=self.kellminmax[1])
        # creating cosine taper to be used
        self.taper, self.w2 = maps.get_taper_deg(self.shape, self.wcs, taper_width_degrees=taperwdeg)
        self.pix_res_arcmin = pix_res_arcmin
        self.N_pix_side = int(m_width_deg / (pix_res_arcmin / 60))
        print(f"Image Size is {self.N_pix_side}")
        # other necessary variables that will be defined when a method is run
        self.n2d = None
        self.n2p = None
        self.kbeam = None

    def tautauTheorycl(self, binned_theoryspec, binned_theory_ells, des_lmin, des_lmax):
        lvals = np.arange(des_lmin, des_lmax)
        f_tautau = interp1d(binned_theory_ells, binned_theoryspec, kind='quadratic', fill_value='extrapolate')
        self.cltautau = f_tautau(lvals)
        self.cltautau *= (2.*np.pi/lvals/(lvals+1.))
        self.theory.loadGenericCls(lvals, self.cltautau, 'tautau', lpad=9000)
        self.incl_tau = True
        return self.cltautau

    def cbfringeTheorycl(self, binned_theoryspec, binned_theory_ells, des_lmin, des_lmax):
        # Meant to be in degrees
        lvals = np.arange(des_lmin,  des_lmax)
        func_cbfringe = interp1d(binned_theory_ells, binned_theoryspec, kind='linear', fill_value='extrapolate')
        self.clalphalpha = ((2.*np.pi) * func_cbfringe(lvals)**2) / (lvals**2)
        self.theory.loadGenericCls(lvals, self.clalphalpha, 'alphalpha', lpad=9000)
        self.incl_cbfringe = True
        return self.clalphalpha

    def flat_lens_sim(self, beam_arcmin=0, noise_uk_arcmin=0, noisep="default", pol=True, incl_tau=False, kappa_ps_fac=1, incl_cbfringe=False, basic_TQU=False):
        if noisep == "default":
            noisep = noise_uk_arcmin * np.sqrt(2.)
        else:
            noisep = noisep

        flsims = lensing.FlatLensingSims(
                self.shape, self.wcs, self.theory, beam_arcmin, noise_uk_arcmin,
                noise_e_uk_arcmin=noisep, noise_b_uk_arcmin=noisep, pol=pol, fixed_lens_kappa=None, incl_tau=incl_tau, kappa_ps_fac=kappa_ps_fac, incl_cbfringe=incl_cbfringe, basic_TQU=basic_TQU)

        # for use in quadratic estimators
        self.n2d = np.nan_to_num(flsims.ps_noise[0,0])
        self.n2p = np.nan_to_num(flsims.ps_noise[1,1])
        self.kbeam = flsims.kbeam
        return flsims

    def gen_basic_cambspec(self, en_cosmo_params=None, vary_initial_params=None, quick_base_run=True):
        base_cosmology_params = {"H0": 67.40,
                                 "ombh2": 0.02237,
                                 "omch2": 0.1199,
                                 "neutrino_hierarchy": 'degenerate',
                                 "num_massive_neutrinos": 1,
                                 "mnu": 0.06,
                                 "nnu": 3.046,
                                 "standard_neutrino_neff": 3.046,
                                 "tau": 0.0561,
                                 "Alens": 1.0}

        base_extra_power_params = {"As": 2.100549e-09,
                                   "ns": 0.9659,
                                   "r": 0}
                                   #"r": 0.03}

        base_lmax_params = {"lmax": 9000,
                            "max_eta_k": 100000,
                            "lens_potential_accuracy": 1}

        base_accuracy_params = {"AccuracyBoost": 2.0,
                                "lSampleBoost": 2.0,
                                "lAccuracyBoost": 1.0}

        final_sampled_params={}

        cosmo_param_dicts = [base_cosmology_params, base_extra_power_params, base_lmax_params, base_accuracy_params]

        if not(quick_base_run):
            assert en_cosmo_params is not None, "Must enter camb parameters as a dictionary."
            # Overwriting each base dictionary and its corresponding parameters with what was input.
            for counter, cpdi in enumerate(cosmo_param_dicts):
                #temp_dict = {}
                for key, value in cpdi.items():
                    if key in en_cosmo_params.keys():
                        cosmo_param_dicts[counter][key] = en_cosmo_params[key]
                        #temp_dict[key] = en_cosmo_params[key]
                #cosmo_param_dicts[counter] = temp_dict

        #for key, value in cosmo_param_dicts[0].items():
        #    print(f"{key}, {value}")

        ocosmo_params = camb.CAMBparams(WantScalars=True, WantTensors=True, Want_CMB=True, Want_CMB_Lensing=True, DoLensing=True)

        num_of_tries = 0
        while num_of_tries < 20:
            try:
                if vary_initial_params is None:
                    pass
                else:
                    for key, val in vary_initial_params.items():
                        # Maybe create a special way to define how the params are sampled
                        if key == "r":
                            #base_extra_power_params[key] = 10**(-3.2)#10.0**(np.random.uniform(-6, -1))
                            cosmo_param_dicts[1][key] = 10.0**(np.random.uniform(-1.99, -1))#-3.2, -1))#0.0001#10**(-3.2)
                            final_sampled_params[key] = cosmo_param_dicts[1][key]
                            #print(final_sampled_params[key])
                        else:
                            ret_cosmop_dict = search_dicts_key(cosmo_param_dicts, key)
                            if ret_cosmop_dict is None:
                                print(f"Key {key} not in any parameter dictionary, skipping it.")
                                continue
                            ret_cosmop_dict[key] = np.random.normal(loc=ret_cosmop_dict[key], scale=val)
                            final_sampled_params[key] = ret_cosmop_dict[key]
                #for key, value in cosmo_param_dicts[0].items():
                #    print(f"{key}, {value}")
                ocosmo_params.set_cosmology(**cosmo_param_dicts[0])
            except TypeError:
                num_of_tries += 1
                print(f"Failed to set parameters: Try number {num_of_tries} of 20")
                continue
            break

        ocosmo_params.InitPower.set_params(**cosmo_param_dicts[1])
        ocosmo_params.set_for_lmax(**cosmo_param_dicts[2])
        ocosmo_params.set_accuracy(**cosmo_param_dicts[3])
        ocosmo_params.NonLinear = camb.model.NonLinear_both
        ocosmo_params.NonLinearModel.set_params()#'mead2020', HMCode_A_baryon = 3.13, HMCode_eta_baryon = 0.603)
        spectra_results = camb.get_results(ocosmo_params)

        # Using orphics theory object as a container for the spectra
        loaded_theory = cosmology.loadTheorySpectraFromPycambResults(spectra_results, ocosmo_params, kellmax=base_lmax_params["lmax"], unlensedEqualsLensed=False, useTotal=True,
                                                                     fill_zero=True, get_dimensionless=False)
        self.theory = loaded_theory
        self.varied_cosmo_params = final_sampled_params

        debug = False
        if debug:
            for key, val in final_sampled_params.items():
                print(f"Parameter {key} is {val}")

        return loaded_theory

def cosine_window(N):
    "makes a cosine window for apodizing to avoid edges effects in the 2d FFT" 
    # make a 2d coordinate system
    ones = np.ones(N)
    inds  = (np.arange(N)+.5 - N/2.)/N *np.pi ## eg runs from -pi/2 to pi/2
    X = np.outer(ones,inds)
    Y = np.transpose(X)
  
    # make a window map
    window_map = np.cos(X) * np.cos(Y)
   
    # return the window map
    return(window_map)

def QU2EB(N,pix_size,Qmap,Umap):
    '''Calcalute E, B maps given input Stokes Q, U maps'''

    # Create 2d Fourier coordinate system.                                      
    ones = np.ones(N)
    inds  = (np.arange(N) - N/2.) /(N-1.)
    kX = np.outer(ones,inds) / (pix_size/60. * np.pi/180.)
    kY = np.transpose(kX)
    ang = np.arctan2(kY,kX)

    # Convert to Fourier domain.                                                
    fQ = np.fft.fftshift(np.fft.fft2(Qmap))
    fU = np.fft.fftshift(np.fft.fft2(Umap))

    # Convert Q, U to E, B in Fourier domain.                                   
    fE = fQ * np.cos(2.*ang) + fU * np.sin(2. *ang)
    fB = - fQ * np.sin(2.*ang) + fU * np.cos(2. *ang)

    # Convert E, B from Fourier to real space.                                  
    Tmap = np.zeros((1, N, N))
    Emap = np.reshape(np.real(np.fft.ifft2(np.fft.fftshift(fE))), (1, N, N))
    Bmap = np.reshape(np.real(np.fft.ifft2(np.fft.fftshift(fB))), (1, N, N))

    TEB_result = np.concatenate((Tmap, Emap, Bmap))

    return TEB_result

def get_flsims_maps(cmbmap_opts, flsims, seed_kappa=None, seed_tau=None, seed_cbf=None, seed_cmb=None, pureEB=False, basic_TQU=True):

    out_maps = flsims.get_sim(return_intermediate=True, seed_kappa=seed_kappa, seed_tau=seed_tau, seed_alpha=seed_cbf, seed_cmb=seed_cmb, basic_TQU=basic_TQU)

    # possible map outputs
    base_cmb_maps_returned = ["primordial", "kappa", "lensed", "beamed", "noise_map", "observed"]
    possible_additional_maps = ["primTEB","cbfringe_map", "tau_map"]
    incl_additional_maps = [basic_TQU, cmbmap_opts.incl_cbfringe, cmbmap_opts.incl_tau]
    final_maps2add = zip(possible_additional_maps, incl_additional_maps)

    idx2insert = 1
    for val0, val1 in final_maps2add:
        if val1:
            base_cmb_maps_returned.insert(idx2insert, val0)
            idx2insert += 1

    gmaps = dict(zip(base_cmb_maps_returned, out_maps))

    # apodized phi map
    gmaps["phi"], _ = lensing.kappa_to_phi(gmaps["kappa"], cmbmap_opts.modlmap, return_fphi=True)

    # applying apodization
    #for key, value in gmaps.items():
    #    if key not in ["noise_map", "beamed"]:
    #        gmaps[key] = value * cmbmap_opts.taper

    # FFT to get T,E,B from apod maps
#    intermed_primTEB = penmap.map2harm(gmaps["primordial"], iau=False)
#    gmaps["primTEB"] = penmap.ifft(intermed_primTEB).real

    #intermed_obsTEB = penmap.map2harm(gmaps["observed"], iau=False)
    #gmaps["obsTEB"] = penmap.ifft(intermed_obsTEB).real

    gmaps["obsTEB"] = QU2EB(cmbmap_opts.N_pix_side, cmbmap_opts.pix_res_arcmin, gmaps["observed"][1], gmaps["observed"][2])# * cmbmap_opts.taper
    gmaps["primTEB"] = gmaps["primTEB"].copy()# * cmbmap_opts.taper

    # filtering maps
    kmask_blur = maps.mask_kspace(cmbmap_opts.shape, cmbmap_opts.wcs, lmax=300)
    gmaps["filteredPrimTEB"] = maps.filter_map(penmap.enmap(gmaps["primTEB"].copy(), cmbmap_opts.wcs), kmask_blur)
    gmaps["filteredObsTEB"] = maps.filter_map(penmap.enmap(gmaps["obsTEB"].copy(), cmbmap_opts.wcs), kmask_blur)

    if pureEB:
        # Getting mask
        kmask_blur = maps.mask_kspace(cmbmap_opts.shape, cmbmap_opts.wcs, lmax=300)

        #new_taper = cosine_window(cmbmap_opts.N_pix_side)

        pure_estimator = maps.Purify(cmbmap_opts.shape, cmbmap_opts.wcs, cmbmap_opts.taper)
        pft_maps_prim = pure_estimator.lteb_from_iqu(gmaps["primordial"], iau=False, flip_q=False)
        pft_maps_obs = pure_estimator.lteb_from_iqu(gmaps["observed"], iau=False, flip_q=False)

        gmaps["purePrimTEB"] = np.zeros((3, cmbmap_opts.N_pix_side, cmbmap_opts.N_pix_side))
        gmaps["pureObsTEB"] = np.zeros((3, cmbmap_opts.N_pix_side, cmbmap_opts.N_pix_side))

        for i in range(3):
            gmaps["purePrimTEB"][i] = (penmap.harm2map(pft_maps_prim[i], iau=False, normalize=True) - np.mean(penmap.harm2map(pft_maps_prim[i], iau=False, normalize=True))) / cmbmap_opts.N_pix_side
            gmaps["pureObsTEB"][i] = (penmap.harm2map(pft_maps_obs[i], iau=False, normalize=True) - np.mean(penmap.harm2map(pft_maps_obs[i], iau=False, normalize=True))) / cmbmap_opts.N_pix_side

        #Filtering
        gmaps["filteredPurePrimTEB"] = maps.filter_map(penmap.enmap(gmaps["purePrimTEB"].copy(), cmbmap_opts.wcs), kmask_blur)
        gmaps["filteredPureObsTEB"] = maps.filter_map(penmap.enmap(gmaps["pureObsTEB"].copy(), cmbmap_opts.wcs), kmask_blur)

    return gmaps

def get_pspectrum(cmbmap_opts, everymap, maptype=["b_prim"]):
    bin_edges = np.arange(39, 651, 36)
    binner = stats.bin2D(cmbmap_opts.modlmap, bin_edges)

    spectrum_holder = {}
    for i in maptype:
        p2d_vals, _, _ = cmbmap_opts.fc.power2d(penmap.ndmap(everymap[i], cmbmap_opts.wcs))
        ells_spec, p1d_vals = binner.bin(p2d_vals)
        spectrum_holder[i+"_spec"] = p1d_vals

    return ells_spec, spectrum_holder

# Getting estimator from notebook
def kendric_method_precompute_window_derivitives(win,pix_size):
    delta = pix_size * np.pi /180. /60.
    dwin_dx =    ((-1.) * np.roll(win,-2,axis =1)      +8. * np.roll(win,-1,axis =1)     - 8. *np.roll(win,1,axis =1)      +np.roll(win,2,axis =1) ) / (12. *delta)
    dwin_dy =    ((-1.) * np.roll(win,-2,axis =0)      +8. * np.roll(win,-1,axis =0)     - 8. *np.roll(win,1,axis =0)      +np.roll(win,2,axis =0) ) / (12. *delta)
    d2win_dx2 =  ((-1.) * np.roll(dwin_dx,-2,axis =1)  +8. * np.roll(dwin_dx,-1,axis =1) - 8. *np.roll(dwin_dx,1,axis =1)  +np.roll(dwin_dx,2,axis =1) ) / (12. *delta)
    d2win_dy2 =  ((-1.) * np.roll(dwin_dy,-2,axis =0)  +8. * np.roll(dwin_dy,-1,axis =0) - 8. *np.roll(dwin_dy,1,axis =0)  +np.roll(dwin_dy,2,axis =0) ) / (12. *delta)
    d2win_dxdy = ((-1.) * np.roll(dwin_dy,-2,axis =1)  +8. * np.roll(dwin_dy,-1,axis =1) - 8. *np.roll(dwin_dy,1,axis =1)  +np.roll(dwin_dy,2,axis =1) ) / (12. *delta)
    return(dwin_dx,dwin_dy,d2win_dx2,d2win_dy2,d2win_dxdy)

def kendrick_method_TQU_to_fourier_TEB(N,pix_size,Tmap,Qmap,Umap,window,dwin_dx,dwin_dy,d2win_dx2,d2win_dy2,d2win_dxdy):
    ### the obvious FFTs
    fft_TxW = np.fft.fftshift(np.fft.fft2(Tmap * window))
    fft_QxW = np.fft.fftshift(np.fft.fft2(Qmap * window))
    fft_UxW = np.fft.fftshift(np.fft.fft2(Umap * window))

    ### the less obvious FFTs that go into the no-leak estiamte
    fft_QxdW_dx = np.fft.fftshift(np.fft.fft2(Qmap * dwin_dx))
    fft_QxdW_dy = np.fft.fftshift(np.fft.fft2(Qmap * dwin_dy))
    fft_UxdW_dx = np.fft.fftshift(np.fft.fft2(Umap * dwin_dx))
    fft_UxdW_dy = np.fft.fftshift(np.fft.fft2(Umap * dwin_dy))
    fft_QU_HOT  = np.fft.fftshift(np.fft.fft2( (2. * Qmap * d2win_dxdy) + Umap * (d2win_dy2 - d2win_dx2) ))
    
    ### generate the polar coordinates needed to cary out the EB-QU conversion
    ones = np.ones(N)
    inds  = (np.arange(N) - N/2.) /(N-1.)
    X = np.outer(ones,inds)
    Y = np.transpose(X)
    R = np.sqrt(X**2. + Y**2. + 1e-9)  ## the small offset regularizes the 1/ell factors below
    ang =  np.arctan2(Y,X)
    ell_scale_factor = 2. * np.pi / (pix_size/60. * np.pi/180.)
    ell2d = R * ell_scale_factor
        
    #p=Plot_CMB_Map(np.real( ang),-np.pi,np.pi,N,N)
    
    
    ### now compute the estimator
    fTmap = fft_TxW
    fEmap = fft_QxW * np.cos(2. * ang) + fft_UxW * np.sin(2. * ang)
    fBmap = (fft_QxW * (-1. *np.sin(2. * ang)) + fft_UxW * np.cos(2. * ang))  ## this line is the nominal B estimator
    fBmap = fBmap - complex(0,2.) / ell2d * (fft_QxdW_dx * np.sin(ang) + fft_QxdW_dy * np.cos(ang))
    fBmap = fBmap - complex(0,2.) / ell2d * (fft_UxdW_dy * np.sin(ang) - fft_UxdW_dx * np.cos(ang))
    fBmap = fBmap +  ell2d**(-2.) * fft_QU_HOT

    ### return the complex fourier maps in 2d

    ### Returning the real maps instead
    rTmap = np.real( (np.fft.ifft2(np.fft.fftshift(fTmap))))
    rEmap = np.real( (np.fft.ifft2(np.fft.fftshift(fEmap))))
    rBmap = np.real( (np.fft.ifft2(np.fft.fftshift(fBmap))-np.mean(np.fft.ifft2(np.fft.fftshift(fBmap)))))
    return(rTmap,rEmap,rBmap)

def get_mlmaps(gmaps, qest_recon_map=None, save_tau=False, save_cbfringe=False, save_pureEB=False, cmbmap_opts=None):
    gmap_keys = ("primordial", "primTEB", "observed", "obsTEB", "kappa", "phi")#, "purePrimTEB", "pureObsTEB")
    emap_keys = ('i_prim', 'q_prim', 'u_prim', 't_prim','e_prim', 'b_prim',
                'i_obs', 'q_obs', 'u_obs', 't_obs', 'e_obs', 'b_obs', 'tru_kappa',
                'tru_phi')#,'pure_t_prim', 'pure_e_prim','pure','','','')

    everymap = {}
    emap_kcounter = 0
    # doesn't work if primTEB is not 3 maps
    for key in gmap_keys:
        dstructure_shape = np.shape(gmaps[key])
        if len(dstructure_shape) == 2:
            everymap[emap_keys[emap_kcounter]] = np.asarray(gmaps[key])
            emap_kcounter += 1
        else:
            for l in np.arange(dstructure_shape[0]):
                everymap[emap_keys[emap_kcounter]] = np.asarray(gmaps[key][l])
                emap_kcounter += 1

    if qest_recon_map is not None:
        everymap['rec_phi'] = np.asarray(np.flipud(qest_recon_map['recon_phi']))
        everymap['rec_kappa'] = np.asarray(np.flipud(qest_recon_map['recon_kappa']))
        everymap['wf_rec_kappa'] = np.asarray(np.flipud(qest_recon_map['wf_recon_kappa']))

    if save_tau:
        everymap["tru_tau"] = np.asarray(np.flipud(gmaps["tau_map"]))

    if save_cbfringe:
        everymap["tru_cbfringe"] = np.asarray(np.flipud(gmaps["cbfringe_map"]))

    if save_pureEB:
        #assert cmbmap_opts is not None, "Missing cmbmap_opts"

        # Order of names is important here
        need_to_save = ["pure_e", "pure_b"]

        # Saving pure maps
        # start enumerate at 1 instead of 0 if not including T
        for mcount, map_name in enumerate(need_to_save, start=1):
            everymap[map_name+"_prim"] = np.asarray(gmaps["purePrimTEB"][mcount])
            everymap[map_name+"_obs"] = np.asarray(gmaps["pureObsTEB"][mcount])
            everymap["filtered_"+map_name+"_prim"] = np.asarray(gmaps["filteredPurePrimTEB"][mcount])
            everymap["filtered_"+map_name+"_obs"] = np.asarray(gmaps["filteredPureObsTEB"][mcount])
        dwin_dx,dwin_dy,d2win_dx2,d2win_dy2,d2win_dxdy = kendric_method_precompute_window_derivitives(cmbmap_opts.taper,cmbmap_opts.pix_res_arcmin)
        pure_maps = kendrick_method_TQU_to_fourier_TEB(cmbmap_opts.N_pix_side,cmbmap_opts.pix_res_arcmin,everymap["t_prim"],everymap["q_prim"],everymap["u_prim"],cmbmap_opts.taper,dwin_dx,dwin_dy,d2win_dx2,d2win_dy2,d2win_dxdy)
        pure_maps_obs = kendrick_method_TQU_to_fourier_TEB(cmbmap_opts.N_pix_side,cmbmap_opts.pix_res_arcmin,everymap["t_obs"],everymap["q_obs"],everymap["u_obs"],cmbmap_opts.taper,dwin_dx,dwin_dy,d2win_dx2,d2win_dy2,d2win_dxdy)
        everymap["pure_b_prim"] = np.asarray(pure_maps[2])
        everymap["pure_b_obs"] = np.asarray(pure_maps_obs[2])
        everymap["pure_e_obs"] = np.asarray(pure_maps_obs[1])

        for mcount, map_name in enumerate(["e", "b"], start=1):
            everymap["filtered_"+map_name+"_prim"] = np.asarray(gmaps["filteredPrimTEB"][mcount])
            everymap["filtered_"+map_name+"_obs"] = np.asarray(gmaps["filteredObsTEB"][mcount])

    return everymap

def create_map_sets(SimSetupCMB_obj, cmbmaps, numbmaps, patchy_tau_vals=None , patchy_tau_ells=None,
                    beam_am=0, noise_uk_am=0, incl_tau=False, incl_cbfringe=False, incl_pureTEB=False, kappa_ps_fac=1,
                    seed_kappa=None, seed_tau=None, seed_cbfringe=None, alpha_ps_vals=None, alpha_ps_ells=None, specmaps=None,
                    gen_spec=False, gen_spec_options={"quick_base_run": True}, load_orph_theories=None, debug=False):
    init_image_shape = (numbmaps,) + SimSetupCMB_obj.shape[-2:]
    init_spectrum_shape = (numbmaps,) + (16,) # hard coded: CHANGE

    poss_maps = (
            't_obs', 'q_obs', 'u_obs', 'e_obs', 'b_obs',
            't_prim', 'q_prim', 'u_prim', 'e_prim', 'b_prim',
            'tru_kappa', 'tru_phi', 'rec_kappa', 'rec_phi',
            'wf_rec_kappa', 'tru_tau', 'tru_cbfringe', 'pure_b_prim', 'pure_b_obs', 'filtered_pure_b_prim', 'filtered_b_prim', 'filtered_pure_e_obs', 'filtered_pure_b_obs', 'pure_e_obs')

    final_maps = {}
    window_vals = {}
    for i in cmbmaps:
        if i not in poss_maps :
            raise Exception('Map {} is not available or in the incorrect format'.format(i))
        else:
            final_maps[i] = np.zeros(init_image_shape, dtype=np.float32)
            final_maps[i+"_spec"] = np.zeros(init_spectrum_shape, dtype=np.float32)

    # initializing theory and param save objects
    saved_theories = []
    if "vary_initial_params" in gen_spec_options.keys():
        for key, val in gen_spec_options["vary_initial_params"].items():
            final_maps['param_'+key] = np.zeros((numbmaps,), dtype=np.float32)

    if not(incl_tau):
        assert 'tru_tau' not in final_maps, "Remove 'tru_tau' from list of maps to save."

    # ADDING THIS VARIABLE
    maps_per_spec = numbmaps
    map_counter = theory_counter = 0

    # generating maps and saving them
    for l in np.arange(0, numbmaps):
        # Including 'Null" in 10% of "numbmaps" set.
        if np.random.random() < 0.85:
            tau_ps_fac = 1.0
            finkappa_ps_fac = 1.0
            alpha_ps_fac = 1.0
        else:
            tau_ps_fac = 0.0
            finkappa_ps_fac = 0.0
            alpha_ps_fac = 0.0

        if load_orph_theories is not None:
            if l < orph_size:
                SimSetupCMB_obj = SimSetupCMB(theory_orph=load_orph_theories[l])
            else:
                SimSetupCMB_obj = SimSetupCMB(theory_orph=load_orph_theories[l])

        if gen_spec:
            if map_counter == maps_per_spec:
                map_counter = 0

            if map_counter == 0:
                gbc_results = SimSetupCMB_obj.gen_basic_cambspec(**gen_spec_options)
                saved_theories.append(gbc_results)
                theory_counter += 1
            map_counter += 1

        if incl_tau:
            cltautau = SimSetupCMB_obj.tautauTheorycl(patchy_tau_vals * tau_ps_fac, patchy_tau_ells, 2, 9000)
        if incl_cbfringe:
            clalpha = SimSetupCMB_obj.cbfringeTheorycl(alpha_ps_vals * alpha_ps_fac, alpha_ps_ells, 2, 9000)

        flsims = SimSetupCMB_obj.flat_lens_sim(beam_arcmin=beam_am, noise_uk_arcmin=noise_uk_am, incl_tau=incl_tau, kappa_ps_fac=finkappa_ps_fac, incl_cbfringe=incl_cbfringe, basic_TQU=True)
        gmaps = get_flsims_maps(SimSetupCMB_obj, flsims, seed_kappa=seed_kappa, seed_tau=seed_tau, seed_cbf=seed_cbfringe, pureEB=incl_pureTEB)
        #recon = qest_recon_map(SimSetupCMB_obj, gmaps, est='EB')
        everymap = get_mlmaps(gmaps, qest_recon_map=None, save_tau=incl_tau, save_cbfringe=incl_cbfringe, save_pureEB=incl_pureTEB, cmbmap_opts=SimSetupCMB_obj)

        if specmaps is not None:
            ells_spec_fin, p1d_dict = get_pspectrum(SimSetupCMB_obj, everymap, maptype=specmaps)
            #Saving spectrum                                                    
            for i in specmaps:
                final_maps[i+"_spec"][l] = p1d_dict[i+"_spec"].astype(np.float32)

                if debug:
                    plt.plot(ells_spec_fin, ells_spec_fin*(ells_spec_fin+1)*p1d_dict[i+"_spec"]/ (2*np.pi) / np.asarray(SimSetupCMB_obj.w2), label=i)
                    plt.yscale("log")
                    plt.xscale("log")
                    plt.legend()
                    plt.savefig('resid_debug_spec_final_spec_{}'.format(i))

        # Debugging
        if debug:
            ells_debug = np.arange(2, 5401)
            uclbb = SimSetupCMB_obj.theory.uCl('BB', ells_debug)
            uclee = SimSetupCMB_obj.theory.uCl('EE', ells_debug)
            uclte = SimSetupCMB_obj.theory.uCl('TE', ells_debug)
            ucltt = SimSetupCMB_obj.theory.uCl('TT', ells_debug)
            debug_unlensed_clBB = {"ells": ells_debug, "unlensed_clBB": uclbb, "unlensed_clEE": uclee, "unlensed_clTE": uclte, "unlensed_clTT": ucltt}
            np.savez("debug_unlensed_clBB", **debug_unlensed_clBB)

            bin_edges = np.arange(39, 651, 36)
            binner = stats.bin2D(SimSetupCMB_obj.modlmap, bin_edges)

            ells_test = np.arange(39, 651, 36)

            # Theory B spectra                                                          
            clbb = SimSetupCMB_obj.theory.uCl('BB', ells_test)
            lclbb = SimSetupCMB_obj.theory.lCl('BB', ells_test)

            #Getting true spectra from theory object input.                         
            p2d_debug_vals, _, _ = SimSetupCMB_obj.fc.power2d(penmap.ndmap(everymap["pure_b_obs"] - everymap["pure_b_prim"], SimSetupCMB_obj.wcs))
            ells_debug, p1d_debug = binner.bin(p2d_debug_vals)

            plt.plot(ells_debug, ells_debug*(ells_debug+1)*p1d_debug / (2*np.pi), label="Residual power")
            plt.plot(ells_test, ells_test*(ells_test+1)*clbb / (2*np.pi), label="Theory Unlensed")
            plt.plot(ells_test, ells_test*(ells_test+1)*lclbb / (2*np.pi), label="Theory lensed")
            plt.yscale("log")
            plt.xscale("log")
            plt.legend()
            plt.savefig('resid_debug_spec')
            plt.close()

        if specmaps is not None:
            ells_spec_fin, p1d_dict = get_pspectrum(SimSetupCMB_obj, everymap, maptype=specmaps)
            #Saving spectrum
            for i in specmaps:
                final_maps[i+"_spec"][l] = p1d_dict[i+"_spec"].astype(np.float32)

        for i in cmbmaps:
            init_map = everymap[i]

            # saving first couple of maps to serve as a check on data quality
            if l < 3:
                #print(np.shape(init_map))
                plt.imsave('final_map_{}_{}.jpg'.format(i,l), init_map, cmap="RdBu_r")
                #np.save('map_{}_{}'.format(i,l), init_map)
                #if i == 't_obs':
                #    print("The max of the T map is {}".format(np.max(init_map)))

            final_maps[i][l] = init_map.astype(np.float32)

        if gen_spec:
            for key, val in (SimSetupCMB_obj.varied_cosmo_params).items():
                final_maps['param_'+key][l] = val

        for l in range(3):
            plt.imsave('final_map_tile_{}_{}.jpg'.format(i,l), final_maps[i][l], cmap="RdBu_r")

    # saving taper and w2 for window correction
    window_vals['taper'] = np.asarray(SimSetupCMB_obj.taper)
    window_vals['w2'] = np.asarray(SimSetupCMB_obj.w2)
    window_vals['spec_cents'] = np.asarray(ells_spec_fin)

    np.savez('window_info', **window_vals)
    np.savez('map_sets32', **final_maps)

    if gen_spec:
        with open('sampled_theory_arr.pickle', 'wb') as f:
            pickle.dump(saved_theories, f)

if __name__ == "__main__":
    # Setting random seed via argument
    selected_rnd_seed = int(sys.argv[1])
    np.random.seed(selected_rnd_seed)
    print("Random seed set to {}".format(selected_rnd_seed))

    cmbmaps = ['q_obs', 'u_obs', 'q_prim', 'u_prim', 'e_prim', 't_obs',
               'tru_kappa', 'tru_phi', 'tru_tau', 't_prim', 'tru_cbfringe', 'b_prim', 'b_obs', 'e_obs', 'pure_b_prim', 'pure_b_obs', 'filtered_pure_b_prim', 'filtered_b_prim', 'filtered_pure_b_obs', 'filtered_pure_e_obs', 'pure_e_obs']
    datadir = '/scratch/users/eguzman/ccsoftware/quicklens/quicklens/data/cl/planck_wp_highL/planck_lensing_wp_highL_bestFit_20130627'

    # Loading Saved Orph Theories for map production later on.
#    with open(orph_pickled_theories_f, 'rb') as pf:
#        my_loaded_theories = pickle.load(pf)



    ttau_ells = np.array([2.0, 70.0, 170.0, 270.0, 370.0, 470.0, 570.0, 670.0, 770.0, 870.0, 970.0, 1070.0, 1170.0, 1270.0, 1370.0,
                            1470.0, 1570.0, 1670.0, 1770.0, 1870.0, 1970.0, 2500.0, 3000.0, 4000.0, 4500.0, 5000.0, 6000.0, 7000.0, 8000.0, 9000.0, 10000.0])
    ttau_vals = np.array([2.0e-10, 1.3e-6, 4.9e-6, 7.2e-6, 8.5e-6, 8.9e-6, 8.6e-6, 8.2e-6, 7.9e-6, 7.4e-6, 6.8e-6, 6.1e-6, 5.7e-6, 5.1e-6, 4.8e-6,
                            4.3e-6, 4e-6, 3.7e-6, 3.4e-6, 3.0e-6, 2.8e-6, 1.8e-6, 1.2e-6, 5e-7, 3.3e-7, 2.2e-7, 9e-8, 4e-8, 1.7e-8, 7e-9, 3e-9])
    ttau_vals *= 0.0

    alphalpha_ells = np.array([1, 10, 65, 100, 210, 235, 270, 355, 385, 420, 520, 580, 650, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000])
    alphalpha_vals = np.array([0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12])
    alphalpha_vals *= 1.0

    # Setting up dictionary of cosmological params to vary.
    scale_to_vary = 5
    planck_params = {"r": 0.10,
                     "Alens": 1,
                     "H0": 70}
    planck_sigma_params = {"r": 0,
                           #"H0": scale_to_vary*0.54,
                           #"ombh2": scale_to_vary*0.00014,
                           #"omch2": scale_to_vary*0.0012,
                           #"ns": scale_to_vary*0.0041,
                           #"As": scale_to_vary*1.01e-10,
                           #"nnu":scale_to_vary*0.18,
                           #"tau":scale_to_vary*0.0071,
                           "Alens":scale_to_vary*0.02}

    sim_map_settings = SimSetupCMB()

    create_map_sets(
        sim_map_settings, cmbmaps, 5000, patchy_tau_vals=ttau_vals, patchy_tau_ells=ttau_ells, beam_am=1.0, noise_uk_am=0.2, incl_tau=True, incl_cbfringe=True, incl_pureTEB=True,
        alpha_ps_vals=alphalpha_vals, alpha_ps_ells=alphalpha_ells, specmaps=["pure_b_prim", "pure_b_obs", "b_prim"], gen_spec=True)
