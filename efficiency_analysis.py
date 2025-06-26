from torch.profiler import profile, record_function, ProfilerActivity
import json
import torch
from cocom import COCOM, COCOMConfig
from tqdm import tqdm
import os
import tempfile
# set torch seed
torch.manual_seed(42)
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
def compute_gflops(prof):
    total_flops = sum([event.flops for event in prof.key_averages()])
    gflops = total_flops / 1e9  # Convert to GFLOPs
    return gflops


def estimate_size_space(compressed_output):
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        torch.save(compressed_output, tmp.name)
        temp_file_path = tmp.name
    file_size_bytes = os.path.getsize(temp_file_path)
    file_size_mb = file_size_bytes / (1024 * 1024)  # Convert bytes to megabytes

    # Get the path to measure its size later

    # Clean up the temporary file
    os.remove(temp_file_path)

    return file_size_mb



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--decoder_model_name",type=str)
    parser.add_argument("--compressor_model_name",type=str, default=None)
    parser.add_argument("--generation_top_k_docs",type=int, default = 5)
    parser.add_argument("--compression_rate",type=int)
    parser.add_argument("--batch_size",type=int, default=16)
    parser.add_argument("--query_length",type=int, default=11)
    parser.add_argument("--label_length",type=int, default=6)
    parser.add_argument("--doc_length",type=int, default=128)
    parser.add_argument("--repeat",type=int, default=10)
    parser.add_argument("--test_mode", type=str, default="generation", choices=["compression", "generation"])
    parser.add_argument("--actual_num", type=int, default=24853658)
    args = parser.parse_args()  

    # first load model
    cfg = COCOMConfig(
    decoder_model_name=args.decoder_model_name,
    quantization='no',
    generation_top_k=args.generation_top_k_docs,
    sep=False,
    compr_model_name=args.compressor_model_name,
    compr_rate=args.compression_rate, 
    compr_linear_type="concat",
    lora=False,
    )
    print("Loading model with config", cfg)

    model = COCOM(cfg)
    model.eval()
    device = torch.device("cuda")
    model = model.to(device)
    print("Model loaded")
    model.decoder.use_cache = False


    print("Preparing input")
    # then prepare input
    if args.test_mode=="compression":
        if model.compr is None:
            # in cocom, we need to add the memory tokens to the input
            total_length = args.doc_length + 2 +  (args.doc_length // args.compression_rate)
        else:
            total_length = args.doc_length
            model.compr.model.use_cache = False
    
    elif args.test_mode=="generation":
        if args.compression_rate > 0:
            num_mem_tokens = (args.doc_length // args.compression_rate ) * args.generation_top_k_docs
            print('num_mem_tokens', num_mem_tokens)
            total_length = args.query_length + ( num_mem_tokens + args.generation_top_k_docs )  + 4 # * 2 because of sep tokens, +4 because bos token, and ['INST'] is 3 tokens
        else:
            total_length = args.query_length  + ( args.doc_length * args.generation_top_k_docs ) + 4
        print('total input length', total_length)


    
    if args.test_mode=="compression":
        if model.compr is None:
            dec_input_ids = torch.randint(0, model.decoder.vocab_size-4, (args.batch_size, total_length), device=device)
            # append the memory tokens to the input
            num_mem = args.doc_length // args.compression_rate
            mem_token_id = model.decoder_tokenizer.mem_token_id
            # the last few tokens are the memory tokens
            dec_input_ids[:, -num_mem:] = mem_token_id
        else:
            dec_input_ids = torch.randint(0, model.compr.model.config.vocab_size-4, (args.batch_size, total_length), device=device)

        attention_mask = torch.ones_like(dec_input_ids, device=device)
        print("size of input", dec_input_ids.size())   
        print("size of attention mask", attention_mask.size())  
    else:
        dec_input_ids = torch.randint(0, model.decoder.vocab_size, (args.batch_size, total_length), device=device)
        
    
    print("Input prepared")

    if args.test_mode == "compression":
        with torch.inference_mode():
            # warm up
            print("started compressing")
            # two cases, depending on if compr is none or not
            if model.compr is None:
                # case decoder only
                compressed_output = model.compr_decoder(input_ids=dec_input_ids, attention_mask=attention_mask)
            else:
                compressed_output = model.compr(input_ids=dec_input_ids, attention_mask=attention_mask)
            # Reset peak memory stats before each run
            disk_space = estimate_size_space(compressed_output)
            actual_space = (args.actual_num * (disk_space/args.batch_size)) / 1024
            print(f"Estimated disk space: {actual_space} GB")
            

            torch.cuda.reset_peak_memory_stats(device)
            with torch.profiler.profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                profile_memory=True,
                with_flops=True 
                ) as prof:
                    with record_function("model_compression"):
                        # Profiling loop
                        for _ in tqdm(range(args.repeat)):
                            if model.compr is None:
                                # case decoder only
                                compressed_output = model.compr_decoder(input_ids=dec_input_ids, attention_mask=attention_mask)
                            else:
                                compressed_output = model.compr(input_ids=dec_input_ids, attention_mask=attention_mask)
            
            for event in prof.key_averages():
                if event.key == 'model_compression':
                    model_compression_event = event
                    break
            gflops = compute_gflops(prof)
            total_peak_memory_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
            # # Compute averages
            print(model_compression_event)

            average_gpu_time_ms = model_compression_event.cuda_time_total / 1e3 / args.repeat
            actual_gpu_time_over_all_hours = (args.actual_num * (average_gpu_time_ms/args.batch_size))/ 1000 / 60/ 60 

            
            total_gflops = gflops 
            average_gflops = gflops / args.repeat
            profiler_output =  {
                    "decoder_model_name": args.decoder_model_name,
                    "compressor_model_name": args.compressor_model_name, 
                    "batch_size": args.batch_size,
                    "generation_top_k_docs": args.generation_top_k_docs,
                    "compression_rate": args.compression_rate,
                    "query_length": args.query_length,
                    # measured units
                    "total_peak_memory_gb": round(total_peak_memory_gb, 1),
                    "average_gpu_time_ms": average_gpu_time_ms,
                    "actual_gpu_time_over_all_hours": actual_gpu_time_over_all_hours,
                    "total_gflops": round(total_gflops, 0),
                }
            print(json.dumps(profiler_output,indent=4))
            
    elif args.test_mode == "generation":
        with torch.inference_mode():
            # warmup 
            inputs_embeds = model.decoder.get_input_embeddings()(dec_input_ids)    
            _ = model.decoder.generate(
                inputs_embeds=inputs_embeds, 
                do_sample=False,
                top_p=None,
                max_new_tokens=args.label_length,
                )
            # Reset peak memory stats before each run
            torch.cuda.reset_peak_memory_stats(device)
            with torch.profiler.profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                profile_memory=True,
                with_flops=True 
                ) as prof:
                    with record_function("model_inference"):
                        # Profiling loop
                        for _ in tqdm(range(args.repeat)):
                            
                            inputs_embeds = model.decoder.get_input_embeddings()(dec_input_ids)
                            input_length = inputs_embeds.size(1)   
                            _ = model.decoder.generate(
                                inputs_embeds=inputs_embeds, 
                                do_sample=False,
                                top_p=None,
                                #ßmax_new_tokens=args.label_length,
                                min_length=args.label_length+input_length,
                                max_length=args.label_length+input_length,
                                )

            for event in prof.key_averages():
                if event.key == 'model_inference':
                    model_inference_event = event
                    break
            gflops = compute_gflops(prof)

            # # Update accumulators
            total_peak_memory_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
            # # Compute averages
            print(model_inference_event)
            #average_cpu_time_ms = model_inference_event.cpu_time_total / 1e3 / args.repeat
            average_gpu_time_ms = model_inference_event.cuda_time_total / 1e3 / args.repeat 
            average_gflops = gflops / args.repeat 


            profiler_output =  {
                    "decoder_model_name": args.decoder_model_name,
                    "compressor_model_name": args.compressor_model_name, 
                    "batch_size": args.batch_size,
                    "generation_top_k_docs": args.generation_top_k_docs,
                    "compression_rate": args.compression_rate,
                    "query_length": args.query_length,
                    # measured units
                    "total_peak_memory_gb": round(total_peak_memory_gb, 1),
                    "average_gpu_time_ms": round(average_gpu_time_ms, 0) ,
                    "average_gflops_per_token": round(average_gflops/args.label_length, 0),
                }

            
            print(json.dumps(profiler_output,indent=4))




        
