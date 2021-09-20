#!/usr/bin/env python3
# Produces various CMB map realizations. Requires orphics package installation
# with all associated dependencies.

from __future__ import division, print_function

import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from orphics import maps, io, stats, cosmology, lensing
from pixell import enmap as penmap

class SimSetupCMB:
    """
    A class used as a container for information needed for producing CMB
    simulations
    """
    def __init__(self, cambRootfile, m_width_deg=5, pix_res_arcmin=2.34375, taperwdeg=1.5):
        """
        Parameters
        ----------
            cambRootfile :
                Filepath to output of CAMB results (specta).
            m_width_deg :
                Width of simulation map in degrees.
            pix_res_arcmin :
                Pixel size in arcmin.
            taperwdeg :
                Width of taper applied to all simulated maps in degrees.
        """
        # map geometry
        self.shape, self.wcs = maps.rect_geometry(width_deg=m_width_deg, px_res_arcmin=pix_res_arcmin)
        self.shape = (3,)+ self.shape
        self.modlmap = penmap.modlmap(self.shape, self.wcs)
        self.fc = maps.FourierCalc(self.shape, self.wcs)
        # theory spectra
        self.theory = cosmology.loadTheorySpectraFromCAMB(
            cambRootfile, unlensedEqualsLensed=False, useTotal=False, TCMB=2.7255e6, lpad=9000, get_dimensionless=False)
        self.cltautau = None
        # map masks
        self.tellminmax = [300, 3500]
        self.pellminmax = [300, 3500]
        self.kellminmax = [20, 3500]
        self.tmask = maps.mask_kspace(self.shape, self.wcs, lmin=self.tellminmax[0], lmax=self.tellminmax[1])
        self.pmask = maps.mask_kspace(self.shape, self.wcs, lmin=self.pellminmax[0], lmax=self.pellminmax[1])
        self.kmask = maps.mask_kspace(self.shape, self.wcs, lmin=self.kellminmax[0], lmax=self.kellminmax[1])
        # cosine taper
        self.taper, self.w2 = maps.get_taper_deg(self.shape, self.wcs, taper_width_degrees=taperwdeg)
        # variables for QE
        self.n2d = None
        self.n2p = None
        self.kbeam = None

    def tautauTheorycl(self, binned_theoryspec, binned_theory_ells, des_lmin, des_lmax):
        """
        Interpolates a binned patchy reionization spectrum and adds it to the orphics theory object.

        Parameters
        ----------
        binned_theoryspec :
            Patchy reionization spectrum in numpy array.
        binned_theory_ells:
            \ell values that correspond to the patchy reionization values.
            Must be in the correct order to match binned_theoryspec values.
        des_lmin : int
            Minimum \ell value interpolation will stop at.
        des_lmax : int
            Maximum \ell value interpolation will stop at.
        """
        lvals = np.arange(des_lmin, des_lmax)
        f_tautau = interp1d(binned_theory_ells, binned_theoryspec, kind='quadratic', fill_value='extrapolate')
        self.cltautau = f_tautau(lvals)
        self.cltautau *= (2.*np.pi/lvals/(lvals+1.))
        self.theory.loadGenericCls(lvals, self.cltautau, 'tautau', lpad=9000)
        return self.cltautau

    def cbfringeTheorycl(self, binned_theoryspec, binned_theory_ells, des_lmin, des_lmax):
        # Meant to be in degrees
        lvals = np.arange(des_lmin,  des_lmax)
        func_cbfringe = interp1d(binned_theory_ells, binned_theoryspec, kind='linear', fill_value='extrapolate')
        self.clalphalpha = ((2.*np.pi) * func_cbfringe(lvals)**2) / (lvals**2)
        self.theory.loadGenericCls(lvals, self.clalphalpha, 'alphalpha', lpad=9000)
        return self.clalphalpha 

    def flat_lens_sim(self, beam_arcmin=0, noise_uk_arcmin=0, noisep="default", pol=True, incl_tau=False, kappa_ps_fac=1, incl_cbfringe=False):
        """
        Creates orphics object used to produce flat-sky simulations.

        Parameters
        ----------
        beam_arcmin :
            Experiment beam FWHM in arcmin
        noise_uk_arcmin :
            Temperature instrument noise in (\mu)K-arcmin
        noisep :
            Equation used for polarization noise.
        incl_tau : boolean or any true object
            If True, observation simulations will be modulated by a patchy reionization map.
        kappa_ps_fac :
            Factor that will be applied to the \kappa power spectrum.
        """
        if noisep == "default":
            noisep = noise_uk_arcmin * np.sqrt(2.)
        else:
            noisep = noisep

        flsims = lensing.FlatLensingSims(
                self.shape, self.wcs, self.theory, beam_arcmin, noise_uk_arcmin,
                noise_e_uk_arcmin=noisep, noise_b_uk_arcmin=noisep, pol=pol, fixed_lens_kappa=None, incl_tau=incl_tau, kappa_ps_fac=kappa_ps_fac, incl_cbfringe=incl_cbfringe)

        # for use in quadratic estimators
        self.n2d = np.nan_to_num(flsims.ps_noise[0,0])
        self.n2p = np.nan_to_num(flsims.ps_noise[1,1])
        self.kbeam = flsims.kbeam
        return flsims

def get_flsims_maps(cmbmap_opts, flsims, incltau=False, inclcbfringe=False, seed_kappa=None, seed_tau=None, seed_cbf=None):
    """
    Generate single set of flat-sky CMB simulations from orphics flsims object,

    Parameters
    ----------
    cmbmap_opts :
        msett class object.
    flsims :
        orphics flsims object for generating maps.
    incltau : boolean
        If True, patchy reionization map will be generated and used to modulate
        primoridial maps by e^{-\\tau} before being lensed.
    seed_kappa : int
        Lensing convergence map NumPy seed.
    seed_tau : int
        Patchy reionization map NumPy seed.

    Returns
    -------
    gmaps :
        A dictionary of all the generated simulation maps.
    """
    out_maps = flsims.get_sim(return_intermediate=True, tauincl=incltau, cbfringe=inclcbfringe, seed_kappa=seed_kappa, seed_tau=seed_tau, seed_cbf=seed_cbf)

    # possible map outputs
    base_cmb_maps_returned = ["primordial", "kappa", "lensed", "beamed", "noise_map", "observed"]
    if incltau and inclcbfringe==False:
        base_cmb_maps_returned.insert(1, "tau_map")
    elif inclcbfringe and incltau==False:
        base_cmb_maps_returned.insert(1, "cbfringe_map")
    elif inclcbfringe and incltau:
        base_cmb_maps_returned.insert(1, "tau_map")
        base_cmb_maps_returned.insert(2, "cbfringe_map")

    gmaps = dict(zip(base_cmb_maps_returned, out_maps))
    
    # apodized phi map
    gmaps["phi"], _ = lensing.kappa_to_phi(gmaps["kappa"], cmbmap_opts.modlmap, return_fphi=True)

    # applying apodization
    for key, value in gmaps.items():
        if key not in ["noise_map", "beamed"]:
            gmaps[key] = value * cmbmap_opts.taper

    # FFT to get T,E,B from apod maps
    intermed_primTEB = penmap.map2harm(gmaps["primordial"], iau=True)
    gmaps["primTEB"] = penmap.ifft(intermed_primTEB).real

    intermed_obsTEB = penmap.map2harm(gmaps["observed"], iau=True)
    gmaps["obsTEB"] = penmap.ifft(intermed_obsTEB).real
    return gmaps

def qest_recon_map(cmbmap_opts, gmaps, pol=True, est='EB'):
    """
    Produces single QE for input realizations from get_flsims_maps function.

    Parameters
    ----------
    cmbmap_opts :
        msett class object.
    gmaps :
        A dictionary of all the generated simulation maps.
    pol : boolean
        If true, will run with polarization simulations
    est : str
        Two letter estimator from T,E,B options

    Returns
    -------
    recon :
        A dictionary with the reconstructed \phi, \kappa, and Wiener filtered
        \kappa.
    """
    # initialize QE
    qest = lensing.qest(
        cmbmap_opts.shape, cmbmap_opts.wcs, cmbmap_opts.theory, noise2d=cmbmap_opts.n2d, beam2d=cmbmap_opts.kbeam,
        kmask=cmbmap_opts.tmask, noise2d_P=cmbmap_opts.n2p, kmask_P=cmbmap_opts.pmask, kmask_K=cmbmap_opts.kmask, pol=pol,
        grad_cut=None, unlensed_equals_lensed=False, bigell=9000)

    # FFT power spectra
    kmapTEB, kmapTEB1, kmapTEB2 = cmbmap_opts.fc.power2d(gmaps['observed'])

    unf_recon_kappa = qest.kappa_from_map(est, kmapTEB1[0], kmapTEB1[1], kmapTEB1[2], alreadyFTed=True)
    recon_phi, _ = lensing.kappa_to_phi(unf_recon_kappa, cmbmap_opts.modlmap, return_fphi=True)

    # WF lensing convergence map
    nlkk2d = qest.N.Nlkk[est]
    clkk2d = qest.N.clkk2d
    wf_intermed = np.nan_to_num(clkk2d / (clkk2d + nlkk2d))

    fftreconKappa = penmap.fft(unf_recon_kappa)
    wf_recon_kappa = penmap.ifft(wf_intermed * fftreconKappa).real

    recon = {"recon_kappa":unf_recon_kappa,"wf_recon_kappa":wf_recon_kappa, "recon_phi":recon_phi}
    return recon


def get_mlmaps(gmaps, qest_recon_map=None, incltau=False, inclcbfringe=False):
    """
    Converts maps produced from get_flsims_maps into numpy arrays.

    Parameters
    ----------
    gmaps :
        Dictionary of maps which must be from get_flsims_maps function or be in
        the same format.
    qest_recon_map :
        Dictionary of QE maps. Must be from qest_recon_map function or in the
        same format.
    incltau : boolean
        If True, it will be assumed tau_map is in the gmaps dictionary.

    Returns
    -------
    amaps :
        Dictionary of all CMB simulation maps. Each map is a numpy array and ready
        for treatment as image data.
    """

    gmap_keys = ("primordial", "primTEB", "observed", "obsTEB", "kappa", "phi")
    emap_keys = ('i_prim', 'q_prim', 'u_prim', 't_prim','e_prim', 'b_prim',
                'i_obs', 'q_obs', 'u_obs', 't_obs', 'e_obs', 'b_obs', 'tru_kappa',
                'tru_phi')

    everymap = {}
    emap_kcounter = 0
    for key in gmap_keys:
        dstructure_shape = np.shape(gmaps[key])
        if len(dstructure_shape) == 2:
            everymap[emap_keys[emap_kcounter]] = np.asarray(np.flipud(gmaps[key]))
            emap_kcounter += 1
        else:
            for l in np.arange(dstructure_shape[0]):
                everymap[emap_keys[emap_kcounter]] = np.asarray(np.flipud(gmaps[key][l]))
                emap_kcounter += 1

    if qest_recon_map is not None:
        everymap['rec_phi'] = np.asarray(np.flipud(qest_recon_map['recon_phi']))
        everymap['rec_kappa'] = np.asarray(np.flipud(qest_recon_map['recon_kappa']))
        everymap['wf_rec_kappa'] = np.asarray(np.flipud(qest_recon_map['wf_recon_kappa']))

    if incltau:
        everymap["tru_tau"] = np.asarray(np.flipud(gmaps["tau_map"]))

    
    if inclcbfringe:
        everymap["tru_cbfringe"] = np.asarray(np.flipud(gmaps["cbfringe_map"]))
    return everymap

def create_map_sets(SimSetupCMB_obj, cmbmaps, numbmaps, patchy_tau_vals=None , patchy_tau_ells=None,
                    beam_am=0, noise_uk_am=0, incl_tau=False, incl_cbfringe=False, kappa_ps_fac=1,
                    seed_kappa=None, seed_tau=None, seed_cbfringe=None, alpha_ps_vals=None, alpha_ps_ells=None):
    init_image_shape = (numbmaps,) + SimSetupCMB_obj.shape[-2:]
    """
    Creates an image data set of CMB simulations.

    Parameters
    ----------
    msett_obj :
        msett class object. Container for all information for creating CMB
        simulations.
    cmbmaps : list
        Python list of CMB maps that should be saved.
    number : int
        Number of simulations to run.
    beam_am :
        Experiment beam FWHM in arcmin
    noise_uk_arcmin :
        Temperature instrument noise in (\mu)K-arcmin
    incl_tau : boolean
        If True, patchy reionization map will be generated and used to modulate
        primoridial maps by e^{-\\tau} before being lensed.
    kappa_ps_fac :
        Factor that will be applied to the \kappa power spectrum.
    seed_kappa : int
        Lensing convergence map NumPy seed.
    seed_tau : int
        Patchy reionization map NumPy seed.

    Returns
    -------
    window_info.npz :
        Numpy array of applied taper information that is saved to the working
        directory. The taper applied to all maps is saved. A naive correction
        "w2" is also saved. The "w2" value is the mean of the squared cosine taper.
    map_sets32.npz :
        Numpy array of CMB maps that is saved to working directory. Maps saved
        are from the cmbmaps list and are float32.
    """

    poss_maps = (
            't_obs', 'q_obs', 'u_obs', 'e_obs', 'b_obs',
            't_prim', 'q_prim', 'u_prim', 'e_prim', 'b_prim',
            'tru_kappa', 'tru_phi', 'rec_kappa', 'rec_phi',
            'wf_rec_kappa', 'tru_tau', 'tru_cbfringe')

    final_maps = {}
    window_vals = {}
    for i in cmbmaps:
        if i not in poss_maps:
            raise Exception('Map {} is not available or in the incorrect format'.format(i))
        else:
            final_maps[i] = np.zeros(init_image_shape, dtype=np.float32)
    
    if not(incl_tau):
        assert 'tru_tau' not in final_maps, "Remove 'tru_tau' from list of maps to save."

    # generating maps and saving them
    for l in np.arange(0, numbmaps):
        # Including 'Null" in 20% of "numbmaps" set.
        if np.random.random() < 0.8:
            tau_ps_fac = 1.0
            finkappa_ps_fac = kappa_ps_fac
            alpha_ps_fac = 1.0
        else:
            tau_ps_fac = 1.0
            finkappa_ps_fac = 1.0
            alpha_ps_fac = 0.0

        if incl_tau:
            cltautau = SimSetupCMB_obj.tautauTheorycl(patchy_tau_vals * tau_ps_fac, patchy_tau_ells, 2, 9000)
        if incl_cbfringe:
            clalpha = SimSetupCMB_obj.cbfringeTheorycl(alpha_ps_vals * alpha_ps_fac, alpha_ps_ells, 2, 9000)
        flsims = SimSetupCMB_obj.flat_lens_sim(beam_arcmin=beam_am, noise_uk_arcmin=noise_uk_am, incl_tau=incl_tau, kappa_ps_fac=finkappa_ps_fac, incl_cbfringe=incl_cbfringe)
        gmaps = get_flsims_maps(SimSetupCMB_obj, flsims, incltau=incl_tau, inclcbfringe=incl_cbfringe, seed_kappa=seed_kappa, seed_tau=seed_tau, seed_cbf=seed_cbfringe)
        #recon = qest_recon_map(SimSetupCMB_obj, gmaps, est='EB')
        everymap = get_mlmaps(gmaps, qest_recon_map=None, incltau=incl_tau, inclcbfringe=incl_cbfringe)

        for i in cmbmaps:
            init_map = everymap[i]

            # saving first couple of maps to serve as a check on data quality
            if l < 3:
                np.save('map_{}_{}'.format(i,l), init_map)
    
            final_maps[i][l] = init_map.astype(np.float32)
    # saving taper and w2 for window correction
    window_vals['taper'] = np.asarray(SimSetupCMB_obj.taper)
    window_vals['w2'] = np.asarray(SimSetupCMB_obj.w2)

    np.savez('window_info', **window_vals)
    np.savez('map_sets32', **final_maps)


if __name__ == "__main__":

    np.random.seed(1225)

    cmbmaps = ['q_obs', 'u_obs', 'q_prim', 'u_prim', 'e_prim', 't_obs',
               'tru_kappa', 'tru_phi', 'tru_tau', 't_prim', 'tru_cbfringe']
    datadir = '../data/camb_spectra/planck_wp_highL/planck_lensing_wp_highL_bestFit_20130627'

    ttau_ells = np.array([2.0, 70.0, 170.0, 270.0, 370.0, 470.0, 570.0, 670.0, 770.0, 870.0, 970.0, 1070.0, 1170.0, 1270.0, 1370.0,
                            1470.0, 1570.0, 1670.0, 1770.0, 1870.0, 1970.0, 2500.0, 3000.0, 4000.0, 4500.0, 5000.0, 6000.0, 7000.0, 8000.0, 9000.0, 10000.0])
    ttau_vals = np.array([2.0e-10, 1.3e-6, 4.9e-6, 7.2e-6, 8.5e-6, 8.9e-6, 8.6e-6, 8.2e-6, 7.9e-6, 7.4e-6, 6.8e-6, 6.1e-6, 5.7e-6, 5.1e-6, 4.8e-6,
                            4.3e-6, 4e-6, 3.7e-6, 3.4e-6, 3.0e-6, 2.8e-6, 1.8e-6, 1.2e-6, 5e-7, 3.3e-7, 2.2e-7, 9e-8, 4e-8, 1.7e-8, 7e-9, 3e-9])
    ttau_vals *= 1.0

    alphalpha_ells = np.array([1, 10, 65, 100, 210, 235, 270, 355, 385, 420, 520, 580, 650, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000])
    alphalpha_vals = np.array([0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12])
    alphalpha_vals *= 1.0

    sim_map_settings = SimSetupCMB(datadir)

    create_map_sets(
        sim_map_settings, cmbmaps, 70000, patchy_tau_vals=ttau_vals, patchy_tau_ells=ttau_ells, beam_am=0.0, noise_uk_am=0.0, incl_tau=True, incl_cbfringe=True,
        alpha_ps_vals=alphalpha_vals, alpha_ps_ells=alphalpha_ells)
