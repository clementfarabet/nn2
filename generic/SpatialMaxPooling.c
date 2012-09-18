#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/SpatialMaxPooling.c"
#else

static int nn_(SpatialMaxPooling_updateOutput)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor);
  int kW = luaT_getfieldcheckint(L, 1, "kW");
  int kH = luaT_getfieldcheckint(L, 1, "kH");
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");
  THTensor *indices = luaT_getfieldcheckudata(L, 1, "indices", torch_Tensor);
  THTensor *output = luaT_getfieldcheckudata(L, 1, "output", torch_Tensor);

  luaL_argcheck(L, input->nDimension == 3, 2, "3D tensor expected");
  luaL_argcheck(L, input->size[2] >= kW && input->size[1] >= kH, 2, "input image smaller than kernel size");

  // sizes
  long nslices = input->size[0];
  long iheight = input->size[1];
  long iwidth = input->size[2];
  long oheight = (iheight - kH) / dH + 1;
  long owidth = (iwidth - kW) / dW + 1;

  // get contiguous input
  input = THTensor_(newContiguous)(input);

  // resize output
  THTensor_(resize3d)(output, nslices, oheight, owidth);

  // indices will contain i,j locatyions for each output point
  THTensor_(resize4d)(indices, 2, nslices, oheight, owidth);

  // get raw pointers
  real *input_data = THTensor_(data)(input);
  real *output_data = THTensor_(data)(output);
  real *indices_data = THTensor_(data)(indices);

  // compute max pooling for each input slice
  long k;
#pragma omp parallel for private(k)
  for (k = 0; k < nslices; k++) {
    // pointers to slices
    real *input_p = input_data + k*iwidth*iheight;
    real *output_p = output_data + k*owidth*oheight;
    real *indy_p = indices_data + k*owidth*oheight;
    real *indx_p = indices_data + (k+nslices)*owidth*oheight;

    // loop over output
    int i,j;
    for(i = 0; i < oheight; i++) {
      for(j = 0; j < owidth; j++) {
        // local pointers
        real *ip = input_p + i*iwidth*dH + j*dW;
        real *op = output_p + i*owidth + j;
        real *indyp = indy_p + i*owidth + j;
        real *indxp = indx_p + i*owidth + j;

        // compute local max:
      	long maxindex = -1;
      	real maxval = -THInf;
      	long tcntr = 0;
        int x,y;
        for(y = 0; y < kH; y++) {
          for(x = 0; x < kW; x++) {
            real val = *(ip + y*iwidth + x);
            if (val > maxval) {
              maxval = val;
              maxindex = tcntr;
            }
            tcntr++;
          }
        }

        // set output to local max
        *op = maxval;

        // store location of max (x,y)
        *indyp = (int)(maxindex / kW)+1;
        *indxp = (maxindex % kW) +1;
      }
    }
  }

  // cleanup
  THTensor_(free)(input);

  return 1;
}

static int nn_(SpatialMaxPooling_updateGradInput)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor);
  THTensor *gradOutput = luaT_checkudata(L, 3, torch_Tensor);
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");
  THTensor *indices = luaT_getfieldcheckudata(L, 1, "indices", torch_Tensor);
  THTensor *gradInput = luaT_getfieldcheckudata(L, 1, "gradInput", torch_Tensor);

  // get contiguous gradOutput
  gradOutput = THTensor_(newContiguous)(gradOutput);

  // resize
  THTensor_(resizeAs)(gradInput, input);
  THTensor_(zero)(gradInput);

  // sizes
  int ichannels = input->size[0];
  int iheight = input->size[1];
  int iwidth = input->size[2];
  int ochannels = ichannels;
  int oheight = gradOutput->size[1];
  int owidth = gradOutput->size[2];

  // get raw pointers
  real *gradInput_data = THTensor_(data)(gradInput);
  real *gradOutput_data = THTensor_(data)(gradOutput);
  real *indices_data = THTensor_(data)(indices);

  // backprop
  long k;
  for (k = 0; k < input->size[0]; k++) {
    // pointers to slices
    real *gradOutput_p = gradOutput_data + k*owidth*oheight;
    real *gradInput_p = gradInput_data + k*iwidth*iheight;
    real *indy_p = indices_data + k*owidth*oheight;
    real *indx_p = indices_data + (k+ochannels)*owidth*oheight;

    // calculate max points
    int i,j;
    for(i = 0; i < oheight; i++) {
      for(j = 0; j < owidth; j++) {
        // retrieve position of max
       	long maxi = *(indy_p + i*owidth + j) - 1 + i*dH;
       	long maxj = *(indx_p + i*owidth + j) - 1 + j*dW;

        // update gradient
        *(gradInput_p + maxi*iwidth + maxj) += *(gradOutput_p + i*owidth + j);
      }
    }
  }

  // cleanup
  THTensor_(free)(gradOutput);

  return 1;
}

static const struct luaL_Reg nn_(SpatialMaxPooling__) [] = {
  {"SpatialMaxPooling_updateOutput", nn_(SpatialMaxPooling_updateOutput)},
  {"SpatialMaxPooling_updateGradInput", nn_(SpatialMaxPooling_updateGradInput)},
  {NULL, NULL}
};

static void nn_(SpatialMaxPooling_init)(lua_State *L)
{
  luaT_pushmetatable(L, torch_Tensor);
  luaT_registeratname(L, nn_(SpatialMaxPooling__), "nn");
  lua_pop(L,1);
}

#endif
