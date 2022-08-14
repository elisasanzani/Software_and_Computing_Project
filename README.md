# ParticleNet implementation for Radiative Muon Capture in the Mu2e Calorimeter

Repository with an adapted version of the [ParticleNet](https://github.com/hqucms/ParticleNet) DG-CNN used to identify signal photons from Radiative Muon Capture (RMC) events in the Mu2e crystal calorimeter.

## Table of contents

- [The Mu2e experiment and its calorimeter](#mu2e-calo) 
- [Radiative Muon Capture](#rmc)
    - [Dataset used](#data)
- [Particle Net and customization](#pnet-custom)
    - [Custom Particle Net](#pnet)
    - [Training and hyper-parameters](#hyper-par)
- [Model final performance](#performance)
- [Other: XGBoost implementation on a reduced dataset](#XGBoost)

<a name="mu2e-calo"></a>
## The Mu2e experiment and its calorimeter 

Mu2e will search for Charged Lepton Flavour Violation via the conversion process: $\mu^-$ $^{27}$Al $\to e^-$ $^{27}$Al, aiming to improve the current sensitivity on the ratio between the conversion and capture events rates by four orders of magnitude, reaching a sensitivity of $8\times 10^{-17}$ at 90\% CL [1](https://arxiv.org/abs/1901.11099) <br>
A high intensity pulsed muon beam at 10 GHz is stopped on the Al target and the interaction products are analysed by the Mu2e detectors:
1. A high momentum resolution 3 meter long Straw Tube Tracker, made of $\sim2\times10^4$ straws arranged in 36 planes, suppresses the irreducible decay in orbit background. 
2. A pure CsI Crystal Calorimeter complements the tracker information and provides excellent energy and time resolution. The Mu2e calorimeter is formed by two annular disks, each one containing 674 undoped CsI crystals (3.4x3.4x20 cm$^3$). Each CsI crystal is readout by two UV-extended Hamamatsu Silicon Photomultipliers.
3. The entire detector region is surrounded by a Cosmic Ray Veto. 
<figure>
    <img src="./images/mu2e.png">
    <figcaption align="center">Fig. 1 The Mu2e experiment and its crystal calorimeter </figcaption>
</figure>
The conversion signature is a 104.97 MeV electron.



<a name="rmc"></a>
## Radiative Muon Capture
Radiative Muon Capture (RMC) occurs when a muon is absorbed in the target and a photon is emitted: $ \mu^- +Al(27,13) \to \gamma + \nu_{\mu} + Mg(27,12)$ <br>
Near the endpoint, RMC photons represent a background to other Mu2e CLFV searches like $\mu^- \to e^+$
The RMC spectrum has been measured by the TRUMPH collaboration, but high energy tails have low statistics and an independent measurement is required near the endpoint.
<figure>
    <img src="./images/rmc-triumph.png">
    <figcaption align="center">Fig. 2 The TRIUMPH RMC photon spectrum.  </figcaption>
</figure>
Backgrounds are due to: <br>
- Beam, called minimun bias (MNBS) <br>
- Cosmic rays <br>
The cut-based analysis performed has an efficiency of around 70%.

<a name="data"></a>
### Dataset used
The used data are available in [main/data-PNet](https://github.com/elisasanzani/Software_and_Computing_Project/tree/main/data-PNet). 
We are interested in the high-energy region of the spectrum and beam background is too high at low-energy and low radius of the calorimeter disk. A preliminary cut has been applied to the data, requiring: E > 50MeV && R > 480 mm.<br>

Three-components dataset with each sample that represents one cluster in the calorimeter <br>
• Point Cloud (collection of xy coordinates of each cell in the cluster)<br>
• Features (collection of E & T for each cell in the cluster)<br>
• Summary (E-T-R-N of the cluster)<br>
(E = energy (MeV), T = mean time (ns), R = centroid radius with respect to the calorimeter center (mm), N = number of active cells above the Mu2e DAQ threshold (arbitrary))

<a name="pnet-custom"></a>
## Particle Net and customization
ParticleNet [2](https://arxiv.org/pdf/1902.08570.pdf) is a CNN-like deep neural network for jet tagging with particle cloud data.
Jets are represented as an unordered, permutation invariant set of particles. Such representation of a jet as a particle cloud is analogous to the point cloud representation of 3D shapes used in computer vision.
<a name="pnet"></a>
### Custom Particle Net



<a name="hyper-par"></a>
### Training and hyper-parameters

<a name="performance"></a>
## Model final performance

<a name="XGBoost"></a>
## Other: XGBoost implementation on a reduced dataset
