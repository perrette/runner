#!/bin/bash
smooth=5e3
glacier=daugaard-jensen
ext=smoothed.$smooth

cmd="python -m glaciertools.preproc smooth -w $smooth glaciers/$glacier.nc -o glaciers/$glacier.$ext.nc"
echo $cmd
eval $cmd
