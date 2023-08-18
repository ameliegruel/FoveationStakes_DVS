# Stakes of Neuromorphic Visual Attention and Foveation

This GitHub repertory contains both work on neuromorphic visual attention and foveation, implemented using spiking neural networks (SNN) and applied to event-based data. 

## Visual attention

The repertory ```VisualAttention``` includes the work on neuromorphic visual attention introduced in the following articles: 
- **Neuromorphic Event-Based Spatio-temporal Attention using Adaptive Mechanisms** by Amélie Gruel, Antonio Vitale, [Jean Martinet](https://niouze.i3s.unice.fr/jmartinet/en/home/) and [Michele Magno](https://ee.ethz.ch/the-department/people-a-z/person-detail.michele-magno.html) (AICAS 2022 - [pdf](https://ieeexplore.ieee.org/document/9869977))
- **Simultaneous Neuromorphic Selection of Multiple Salient Objects for Event Vision** by Amélie Gruel, [Jean Martinet](https://niouze.i3s.unice.fr/jmartinet/en/home/) and [Michele Magno](https://ee.ethz.ch/the-department/people-a-z/person-detail.michele-magno.html) (IJCNN 2023)

You can find the corresponding README [here](VisualAttention/README.md).

The repertory ```Foveation``` includes the work on neuromorphic foveation introduced in the paper:
- **Stakes of Neuromorphic Foveation: a promising future for embedded event cameras** by Amelie Gruel, Dalia Hareb, [Antoine Grimaldi](https://laurentperrinet.github.io/author/antoine-grimaldi/), [Jean Martinet](https://niouze.i3s.unice.fr/jmartinet/en/home/), [Laurent Perrinet](https://laurentperrinet.github.io), [Bernabe Linares-Barranco](http://www2.imse-cnm.csic.es/~bernabe/) and [Teresa Serrano-Gotarredona](http://www2.imse-cnm.csic.es/~terese/) (Biological Cybernetics 2022 - [pdf](https://www.researchsquare.com/article/rs-2120721/v1)).

You can find the corresponding README [here](Foveation/README.md).

## References

If you use any of this code, please cite the following publications:
```bibtex
@Article{Gruel_AICAS_2022,
	author = {Gruel, Amélie and Vitale, Antonio and Martinet, Jean and Magno, Michele},
	journal = {IEEE International Conference on Artificial Intelligence Circuits and Systems (AICAS)},
	title = {Neuromorphic Event-Based Spatio-temporal Attention using Adaptive Mechanisms},
	year = {2022}
}
```

```bibtex
@Article{Gruel_IJCNN_2023,
  title={Simultaneous neuromorphic selection of multiple salient objects for event vision},
  author={Gruel, Amélie and Martinet, Jean and Magno, Michele},
  journal={2023 International Joint Conference on Neural Networks (IJCNN)},
  year={2023}
}
```

```bibtex
@Article{Gruel_BiologicalCybernetics_2022,
  author        = {Amélie Gruel and Dalia Hareb and Antoine Grimaldi and Jean Martinet and Laurent Perrinet and Bernabe Linares-Barranco and Teresa Serrano-Gotarredona},
  title         = {Stakes of Neuromorphic Foveation: a promising future for embedded event cameras},
  journal       = {Biological Cybernetics},
  url           = {https://doi.org/10.21203/rs.3.rs-2120721/v1},
  year          = 2022,
  note          = {Submitted for the special issue "What can Computer Vision learn from Visual Neuroscience?". PREPRINT (Version 1) available at Research Square}
}
```

## Acknowledgements

This work was supported by the European Union’s ERA-NET CHIST-ERA2018 research and innovation programme under grant agreement ANR-19-CHR3-0008. The authors are grateful to the OPAL infrastructure from Université Côte d’Azur for providing resources and support.
