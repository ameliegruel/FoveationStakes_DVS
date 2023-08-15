# Stakes of Neuromorphic Foveation
This is the code for the paper **Stakes of Neuromorphic Foveation: a promising future for embedded event cameras** by Amelie Gruel, Dalia Hareb, [Antoine Grimaldi](https://laurentperrinet.github.io/author/antoine-grimaldi/), [Jean Martinet](https://niouze.i3s.unice.fr/jmartinet/en/home/), [Laurent Perrinet](https://laurentperrinet.github.io), [Bernabe Linares-Barranco](http://www2.imse-cnm.csic.es/~bernabe/) and [Teresa Serrano-Gotarredona](http://www2.imse-cnm.csic.es/~terese/)

You can find the pdf of the paper [here](https://www.researchsquare.com/article/rs-2120721/v1). If you use any of this code, please cite the following publication:

```bibtex
@Article{Gruel2022,
  author        = {Amélie Gruel and Dalia Hareb and Antoine Grimaldi and Jean Martinet and Laurent Perrinet and Bernabe Linares-Barranco and Teresa Serrano-Gotarredona},
  title         = {Stakes of Neuromorphic Foveation: a promising future for embedded event cameras},
  journal       = {Biological Cybernetics},
  url           = {https://doi.org/10.21203/rs.3.rs-2120721/v1},
  year          = 2022,
  note          = {Submitted for the special issue "What can Computer Vision learn from Visual Neuroscience?". Currently under revision. PREPRINT (Version 1) available at Research Square}
}
```

## Summary of the work

Abstract of the extended article in Biological Cybernetics (*currently under revision*):
> Foveation can be defined as the organic action of directing the gaze towards a visual region of interest to acquire relevant information selectively. With the recent advent of event cameras, we believe that taking advantage of this visual neuroscience mechanism would greatly improve the efficiency of event data processing. Indeed, applying foveation to event data would allow to comprehend the visual scene while significantly reducing the amount of raw data to handle. In this respect, we demonstrate the stakes of neuromorphic foveation theoretically and empirically across several computer vision tasks, namely semantic segmentation and classification. We show that foveated event data has a significantly better trade-off between quantity and quality of the information conveyed than high or low-resolution event data. Furthermore, this compromise extends even over fragmented datasets.

### Main steps of the methodology 

We present below the main steps of the methodology used in this work to demonstrate the stakes of neuromorphic foveation: 
- the event data is spatially downscaled by a dividing factor ρ = 4 before being given as input to the saliency detection model. Each downscaling method described in the previous section is used to produce 4 different sets of event data, on which the saliency is detected separately.
- according to the saliency detected on those 4 “reduced” sets, 8 new sets of event data are created:
  - 4 “only RoIs” sets containing only event data from the high-resolution dataset in the regions detected as salient in the respective “reduced” set;
  – 4 “foveated” sets reconstructed using the binary foveation process described above.
- finally, the classification or semantic segmentation performance is measured on the 12 newly created sets of event data (“reduced”, “only RoIs” and “foveated”) as well as on the original dataset.

## Execution

### Install

### Run

#### Reduce event data

#### Get RoI coordinates

```bash
python run_reconstruction.py \
  -c pretrained/E2VID_lightweight.pth.tar \
  -i data/dynamic_6dof.zip \
  --auto_hdr \
  --display \
  --show_events
```

#### Create “only RoIs” dataset

#### Create “foveated” dataset

### Example datasets 

## See also
Workshop paper from which is extended the extended article: **Neuromorphic foveation applied to semantic segmentation** by Amelie Gruel, Dalia Hareb, [Jean Martinet](https://niouze.i3s.unice.fr/jmartinet/en/home/), [Bernabe Linares-Barranco](http://www2.imse-cnm.csic.es/~bernabe/) and [Teresa Serrano-Gotarredona](http://www2.imse-cnm.csic.es/~terese/])([pdf](https://drive.google.com/file/d/1-r9l4bmoaJe2RIvn30GPa15-yUyXXwp8/view)).

## Acknowledgements

This work was supported by the European Union’s ERA-NET CHIST-ERA2018 research and innovation programme under grant agreement ANR-19-CHR3-0008. The authors are grateful to the OPAL infrastructure from Universit ́e Cˆote d’Azur for providing resources and support.
