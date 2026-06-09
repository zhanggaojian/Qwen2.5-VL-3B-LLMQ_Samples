The folder is used to run on device with genie


1 run script to get embedding and test inputs,  the script will do next work

  1.1 Export llm embedding, the data will be used by genie-t2t-run

  1.2 run visual model on gpu or you can run vit on device to get vision embedding

  1.3 group vision embedding and text embedding together depend on model requirement

  1.4 save the final embedding, the data will be used by genie-t2t-run



2 push libs and binary into device 

​  2.1 QNN libs(libQnnHtp.so, libQnnSystem.so,libQnnHtpNetRunExtensions.so and dsp arch related libs) 

​  2.2 libGenie.so 

  2.3 genie-t2t-run 



3 push model binary, config file into device

  3.1 push llm binary or vit binary into device

​  3.2 change genie config file like qwen2.5vl.json, the value is depend on different models

​  3.3 push tokenizer.json

​  3.4 push htp_backend_ext_config.json, need to change soc_id/dsp_arch based on real target



4 run testing in device

  4.1 setting environment variable like  LD_LIBRARY_PATH/ADSP_LIBRARY_PATH/CDSP_LIBRARY_PATH

  4.2 run genie-t2t-run

```
./genie-t2t-run -c qwen2.5vl.json -e input_embeds.bin -t embedding_weights_151936x2048.raw
```

#TODO

mrope is not supported by Genie now, need to add some codes in genie, it will improve accuracy.

