nn2
===

nn2 is the successor of nn. The main thing we're trying to achieve here is:

  * better consistency across modalities (Volumetric, Spatial, Temporal)
  * better performance by packing features in memory

TODO List
---------

'Spatial' modules need to invert their convention, by packing the features
in memory. Modules affected:

   * SpatialConvolution
   * SpatialConvolutionMap       
   * SpatialMaxPooling
   * SpatialSubSampling
   * Spatial*Normalization
   * SpatialLPPooling
   * SpatialZeroPadding

'Volumetric' modules:

   * VolumetricConvolution

