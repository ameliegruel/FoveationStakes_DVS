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

!! TO UPDATE !!

### Run
Scripts `BinaryFoveation/Automatic[...].py` browse the entirety of the event data in an input directory and apply the corresponding process (downscaling, foveation, saliency detection, etc). 

#### Reduce event data

```bash
python AutomaticReducedData.py \
  --dataset /path/to/dataset/directory/ \
  --divider ρ_value \
  --method downscaling_method
```
The default value of the dividing factor ρ is 4. The downscaling method is to be chosen from the set ['funelling', 'eventcount', 'cubic', 'linear'] and is imported from the script `reduceEvents.py` (see the last section for more details on the different methods available).

#### Get RoI coordinates

```bash
python AutomaticROIData.py \
  --dataset /path/to/dataset/directory/ \
  --divider ρ_value \
  --method downscaling_method
```
The divider and method arguments allow to reconstruct the path where to grab the downscaled data. If the method is "none", this means that the RoIs are detected on the original dataset, at high resolution. Since the neuromorphic foveation is to be constructed from low-resolution events whose resolution is to be sharpened in the salient regions, this option is not used in this work.

The RoIs coordinates are saved as positive events, which <x,y,t> channels provide the spatiotemporal locations of RoIs in the input data. 

#### Create “foveated” dataset

```bash
python AutomaticFoveatedData.py \
  --dataset /path/to/dataset/directory/ \
  --divider ρ_value \
  --method downscaling_method \
  --ROI /path/to/ROI/coordinates/directory/
```

The data is foveated according to a binary mask, thanks to the function `getPuzzle()` imported from the script `getFoveatedData.py`.

### Datasets 

The results provided in the paper were obtained while assessing the neuromorphic foveation on two benchmark event-based datasets: 
- DVS 128 Gesture ([dataset](https://research.ibm.com/interactive/dvsgesture/), [CVPR 2017 paper](https://openaccess.thecvf.com/content_cvpr_2017/papers/Amir_A_Low_Power_CVPR_2017_paper.pdf)) 
- DAVIS Driving Dataset 17 ([code and dataset](https://docs.google.com/document/d/1HM0CSmjO8nOpUeTvmPjopcBcVCk7KXvLUuiZFS6TWSg/pub), [ICML 2017 paper](https://arxiv.org/abs/1711.01458)).

### Computer vision tasks

- Classification: accuracy evaluated using the [HOTS model](https://laurentperrinet.github.io/publication/grimaldi-21-cosyne/)
- Semantic segmentation: accuracy and MIoU evaluated using the [Ev-SegNet model](https://arxiv.org/abs/1811.12039). It is to be noted that the Ev-SegNet model uses as input the DDD17 event data with a specific representation: the data comprises six channels corresponding to the count, mean and standard deviation of the normalised timestamps of events happening at each pixel within an interval of 50ms for the positive and negative polarities. Dalia Hareb identified the data used to train the model in the original dataset (4 channels - <x,y,p,t>) and reconstructed this data in the format taken into account by Ev-SegNet (6 channels). 

## Acknowledgements

This work was supported by the European Union’s ERA-NET CHIST-ERA2018 research and innovation programme under grant agreement ANR-19-CHR3-0008. The authors are grateful to the OPAL infrastructure from Université Côte d’Azur for providing resources and support.

## See also
Workshop paper from which is extended the extended article: **Neuromorphic foveation applied to semantic segmentation** by Amelie Gruel, Dalia Hareb, [Jean Martinet](https://niouze.i3s.unice.fr/jmartinet/en/home/), [Bernabe Linares-Barranco](http://www2.imse-cnm.csic.es/~bernabe/) and [Teresa Serrano-Gotarredona](http://www2.imse-cnm.csic.es/~terese/]) ([pdf](https://drive.google.com/file/d/1-r9l4bmoaJe2RIvn30GPa15-yUyXXwp8/view))

Complete description of the four spatial downscaling methods in the VISAPP 2021 article: **Event data downscaling for embedded computer vision** by Amelie Gruel, [Jean Martinet](https://niouze.i3s.unice.fr/jmartinet/en/home/), [Teresa Serrano-Gotarredona](http://www2.imse-cnm.csic.es/~terese/]) and [Bernabe Linares-Barranco](http://www2.imse-cnm.csic.es/~bernabe/) ([pdf](https://cnrs.hal.science/hal-03814075/))
