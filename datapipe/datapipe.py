#!/usr/bin/env python2
# Produces various CMB map realizations. Requires orphics package installation
# with all associated dependencies.

from __future__ import division, print_function

import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from orphics import maps, io, stats, cosmology, lensing
from pixell import enmap as penmap

class msett:
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

    def flat_lens_sim(self, beam_arcmin=0, noise_uk_arcmin=0, noisep="default", pol=True, incl_tau=None, kappa_ps_fac=1):
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
                noise_e_uk_arcmin=noisep, noise_b_uk_arcmin=noisep, pol=pol, fixed_lens_kappa=None, tautauspec=incl_tau, kappa_ps_fac=kappa_ps_fac)

        # for use in quadratic estimators
        self.n2d = np.nan_to_num(flsims.ps_noise[0,0])
        self.n2p = np.nan_to_num(flsims.ps_noise[1,1])
        self.kbeam = flsims.kbeam
        return flsims

def get_flsims_maps(cmbmap_opts, flsims, incltau=False, seed_kappa=None, seed_tau=None):
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
    out_maps = flsims.get_sim(return_intermediate=True, tauincl=incltau, seed_kappa=seed_kappa, seed_tau=seed_tau)

    # shifting the out_maps list if tauincl is True
    index = 1
    if incltau:
        index += 1

    # getting phi map
    unap_tru_phi, _ = lensing.kappa_to_phi(out_maps[index], cmbmap_opts.modlmap, return_fphi=True)

    # getting TEB maps
    intermed_unlTEB = penmap.map2harm(out_maps[0] * cmbmap_opts.taper, iau=True)
    unlTEB = penmap.ifft(intermed_unlTEB).real

    intermed_lenTEB = penmap.map2harm(out_maps[index + 4] * cmbmap_opts.taper, iau=True)
    lenTEB = penmap.ifft(intermed_lenTEB).real

    kappa = out_maps[index] * cmbmap_opts.taper
    tru_phi = unap_tru_phi * cmbmap_opts.taper
    observed = out_maps[index + 4] * cmbmap_opts.taper
    unlensed = out_maps[0] * cmbmap_opts.taper
    beamed = out_maps[index + 2]
    noise_map = out_maps[index + 3]

    gmaps = {
        'unlensed':unlensed, 'kappa':kappa,
        'beamed':beamed, 'noise_map':noise_map,
        "observed":observed, "unlTEB":unlTEB,
        "lenTEB":lenTEB, "phi":tru_phi}
    if incltau:
        tau_map = out_maps[1] * cmbmap_opts.taper
        gmaps["tau_map"] = tau_map
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
    pTEB, kmapTEB1, kmapTEB2 = cmbmap_opts.fc.power2d(gmaps['observed'])

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


def get_mlmaps(gmaps, qest_recon_map=None, incltau=False):
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

    # flipping for (y,x) pairs
    q_unl = np.asarray(np.flipud(gmaps["unlensed"][1]))
    u_unl = np.asarray(np.flipud(gmaps["unlensed"][2]))
    t_unl = np.asarray(np.flipud(gmaps["unlTEB"][0]))
    e_unl = np.asarray(np.flipud(gmaps["unlTEB"][1]))
    b_unl = np.asarray(np.flipud(gmaps["unlTEB"][2]))
    q_len = np.asarray(np.flipud(gmaps["observed"][1]))
    u_len = np.asarray(np.flipud(gmaps["observed"][2]))
    t_len = np.asarray(np.flipud(gmaps["lenTEB"][0]))
    e_len = np.asarray(np.flipud(gmaps["lenTEB"][1]))
    b_len = np.asarray(np.flipud(gmaps["lenTEB"][2]))
    kappa_tru = np.asarray(np.flipud(gmaps["kappa"]))
    phi_tru = np.asarray(np.flipud(gmaps["phi"]))

    amaps = {
        'q_unl':q_unl, 'u_unl':u_unl,
        't_unl':t_unl, 'e_unl':e_unl,
        'b_unl':b_unl, 'q_len':q_len,
        'u_len':u_len, 't_len':t_len,
        'e_len':e_len, 'b_len':b_len,
        'tru_kappa':kappa_tru, 'tru_phi':phi_tru}

    if qest_recon_map is not None:
        recon_phi = np.asarray(np.flipud(qest_recon_map['recon_phi']))
        recon_kappa = np.asarray(np.flipud(qest_recon_map['recon_kappa']))
        wf_recon_kappa = np.asarray(np.flipud(qest_recon_map['wf_recon_kappa']))
        amaps['rec_kappa'] = recon_kappa
        amaps['rec_phi'] = recon_phi
        amaps['wf_rec_kappa'] = wf_recon_kappa

    if incltau:
        tau_tru = np.asarray(np.flipud(gmaps["tau_map"]))
        amaps["tru_tau"] = tau_tru
    return amaps

def create_map_sets(msett_obj, cmbmaps, number, beam_am=0, noise_uk_am=0, incl_tau=False, kappa_ps_fac=1, seed_kappa=None, seed_tau=None):
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
    # class object of map options to be used for simulations
    mset_opts = msett_obj

    poss_maps = [
            't_len', 'q_len', 'u_len', 'e_len', 'b_len',
            't_unl', 'q_unl', 'u_unl', 'e_unl', 'b_unl',
            'tru_kappa', 'tru_phi', 'rec_kappa', 'rec_phi',
            'wf_rec_kappa', 'tru_tau']

    final_maps = {}
    window_vals = {}
    for i in cmbmaps:
        if i not in poss_maps:
            raise Exception('Map {} is not available or in the incorrect format'.format(i))
        else:
            final_maps[i] = []

    for l in np.arange(0, number):

        if np.random.random() < 0.8:
            cltautau = mset_opts.tautauTheorycl(def_ttau_vals, def_ttau_els, 2, 9000)
            flsims = mset_opts.flat_lens_sim(beam_arcmin=beam_am, noise_uk_arcmin=noise_uk_am, incl_tau=mset_opts.cltautau, kappa_ps_fac=kappa_ps_fac)
        else:
            cltautau = mset_opts.tautauTheorycl(def_ttau_vals * 0.0, def_ttau_els, 2, 9000)
            flsims = mset_opts.flat_lens_sim(beam_arcmin=beam_am, noise_uk_arcmin=noise_uk_am, incl_tau=mset_opts.cltautau, kappa_ps_fac=0.0)
        gmaps = get_flsims_maps(mset_opts, flsims, incltau=incl_tau, seed_kappa=seed_kappa, seed_tau=seed_tau)
        recon = qest_recon_map(mset_opts, gmaps, est='EB')
        amaps = get_mlmaps(gmaps, qest_recon_map=recon, incltau=incl_tau)

        for i in cmbmaps:
            init_map = amaps[i]

            # saving first couple of maps to serve as a check on data quality
            if l < 3:
                np.save('map_{}_{}'.format(i,l), init_map)

            final_maps[i].append(init_map)

    for i in cmbmaps:
        final_maps[i] = np.asarray(final_maps[i], dtype=np.float32)

    window_vals['taper'] = np.asarray(mset_opts.taper)
    window_vals['w2'] = np.asarray(mset_opts.w2)

    np.savez('window_info', **window_vals)
    np.savez('map_sets32', **final_maps)


if __name__ == "__main__":

    np.random.seed(1225)

    cmbmaps = ['q_len', 'u_len', 'q_unl', 'u_unl', 'e_unl', 'tru_kappa', 'rec_kappa', 'tru_phi', 'rec_phi', 'wf_rec_kappa', 'tru_tau', 't_len', 't_unl']
    datadir = '../data/camb_spectra/planck_wp_highL/planck_lensing_wp_highL_bestFit_20130627'

    def_ttau_els = np.array([2.0, 70.0, 170.0, 270.0, 370.0, 470.0, 570.0, 670.0, 770.0, 870.0, 970.0, 1070.0, 1170.0, 1270.0, 1370.0, 1470.0, 1570.0, 1670.0, 1770.0, 1870.0, 1970.0, 2500.0, 3000.0, 4000.0, 4500.0, 5000.0, 6000.0, 7000.0, 8000.0, 9000.0, 10000.0])
    def_ttau_vals = np.array([2.0e-10, 1.3e-6, 4.9e-6, 7.2e-6, 8.5e-6, 8.9e-6, 8.6e-6, 8.2e-6, 7.9e-6, 7.4e-6, 6.8e-6, 6.1e-6, 5.7e-6, 5.1e-6, 4.8e-6, 4.3e-6, 4e-6, 3.7e-6, 3.4e-6, 3.0e-6, 2.8e-6, 1.8e-6, 1.2e-6, 5e-7, 3.3e-7, 2.2e-7, 9e-8, 4e-8, 1.7e-8, 7e-9, 3e-9])
    def_ttau_vals *= 1.0

    mset_opts = msett(datadir)
    cltautau = mset_opts.tautauTheorycl(def_ttau_vals, def_ttau_els, 2, 9000)

    create_map_sets(mset_opts, cmbmaps, 70000, beam_am=0.0, noise_uk_am=0.0, incl_tau=True)
