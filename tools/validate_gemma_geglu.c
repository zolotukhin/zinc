/* Remote GPU only: native Q4_K quantization, output/stride guards and paired
 * bitwise gate/up/GEGLU proof. No inference or whole-model performance claim.
 * Build like validate_q8_decode.c. Intel assets have separate names; generic shaders remain the oracle. */
#define main shared_cache_fixture_main
#include "validate_dmmv_shared_cache.c"
#undef main

typedef struct { uint32_t M, N, K, stride_b, stride_d, a_offset, b_offset, d_offset; } GemmPush;
static uint32_t pairs;
static double maximum_absolute_error;

static void dispatch_geglu(const char *name, Buffer *b, GemmPush push) {
    VkDescriptorSetLayoutBinding bindings[4];
    for (uint32_t i=0;i<4;i++) bindings[i]=(VkDescriptorSetLayoutBinding){
        .binding=i,.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,.descriptorCount=1,.stageFlags=VK_SHADER_STAGE_COMPUTE_BIT};
    VkDescriptorSetLayoutCreateInfo li={.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,.bindingCount=4,.pBindings=bindings};
    VkDescriptorSetLayout set_layout;check(vkCreateDescriptorSetLayout(device,&li,NULL,&set_layout));
    VkPushConstantRange range={.stageFlags=VK_SHADER_STAGE_COMPUTE_BIT,.size=sizeof push};
    VkPipelineLayoutCreateInfo pi={.sType=VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,.setLayoutCount=1,.pSetLayouts=&set_layout,.pushConstantRangeCount=1,.pPushConstantRanges=&range};
    VkPipelineLayout layout;check(vkCreatePipelineLayout(device,&pi,NULL,&layout));
    VkShaderModule module=shader(name);
    VkComputePipelineCreateInfo ci={.sType=VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage={.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,.stage=VK_SHADER_STAGE_COMPUTE_BIT,.module=module,.pName="main"},.layout=layout};
    VkPipeline pipeline;check(vkCreateComputePipelines(device,VK_NULL_HANDLE,1,&ci,NULL,&pipeline));
    VkDescriptorPoolSize pool_size={.type=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,.descriptorCount=4};
    VkDescriptorPoolCreateInfo pool_info={.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,.maxSets=1,.poolSizeCount=1,.pPoolSizes=&pool_size};
    VkDescriptorPool pool;check(vkCreateDescriptorPool(device,&pool_info,NULL,&pool));
    VkDescriptorSetAllocateInfo si={.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,.descriptorPool=pool,.descriptorSetCount=1,.pSetLayouts=&set_layout};
    VkDescriptorSet set;check(vkAllocateDescriptorSets(device,&si,&set));
    VkDescriptorBufferInfo infos[4];VkWriteDescriptorSet writes[4];
    for(uint32_t i=0;i<4;i++) {
        infos[i]=(VkDescriptorBufferInfo){.buffer=b[i].handle,.range=b[i].size};
        writes[i]=(VkWriteDescriptorSet){.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,.dstSet=set,.dstBinding=i,.descriptorCount=1,.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,.pBufferInfo=&infos[i]};
    }
    vkUpdateDescriptorSets(device,4,writes,0,NULL);
    VkCommandBufferAllocateInfo ai={.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,.commandPool=command_pool,.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY,.commandBufferCount=1};
    VkCommandBuffer command;check(vkAllocateCommandBuffers(device,&ai,&command));
    VkCommandBufferBeginInfo begin={.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};check(vkBeginCommandBuffer(command,&begin));
    VkMemoryBarrier before={.sType=VK_STRUCTURE_TYPE_MEMORY_BARRIER,.srcAccessMask=VK_ACCESS_HOST_WRITE_BIT,.dstAccessMask=VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT};
    vkCmdPipelineBarrier(command,VK_PIPELINE_STAGE_HOST_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,1,&before,0,NULL,0,NULL);
    vkCmdBindPipeline(command,VK_PIPELINE_BIND_POINT_COMPUTE,pipeline);
    vkCmdBindDescriptorSets(command,VK_PIPELINE_BIND_POINT_COMPUTE,layout,0,1,&set,0,NULL);
    vkCmdPushConstants(command,layout,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof push,&push);
    vkCmdDispatch(command,(push.M+31)/32,(push.N+31)/32,1);
    VkMemoryBarrier after={.sType=VK_STRUCTURE_TYPE_MEMORY_BARRIER,.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT,.dstAccessMask=VK_ACCESS_HOST_READ_BIT};
    vkCmdPipelineBarrier(command,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_HOST_BIT,0,1,&after,0,NULL,0,NULL);
    check(vkEndCommandBuffer(command));
    VkSubmitInfo submit={.sType=VK_STRUCTURE_TYPE_SUBMIT_INFO,.commandBufferCount=1,.pCommandBuffers=&command};
    check(vkQueueSubmit(queue,1,&submit,VK_NULL_HANDLE));check(vkQueueWaitIdle(queue));
    vkFreeCommandBuffers(device,command_pool,1,&command);
    vkDestroyDescriptorPool(device,pool,NULL);vkDestroyPipeline(device,pipeline,NULL);vkDestroyShaderModule(device,module,NULL);
    vkDestroyPipelineLayout(device,layout,NULL);vkDestroyDescriptorSetLayout(device,set_layout,NULL);
}

static void paired_geglu(uint32_t M,uint32_t N,uint32_t K,int full,uint32_t salt) {
    const uint32_t a_offset=32,b_offset=7,d_offset=5,stride_b=K+11,stride_d=M+9;
    const size_t row_bytes=ggml_row_size(GGML_TYPE_Q4_K,K);
    const size_t count=d_offset+(size_t)N*stride_d+13;
    Buffer b[4]={buffer(a_offset+(size_t)M*row_bytes+32),buffer(a_offset+(size_t)M*row_bytes+32),
        buffer((b_offset+(size_t)N*stride_b+13)*sizeof(float)),buffer(count*sizeof(float))};
    float *row=allocate(K*sizeof(float));float *decoded[2]={allocate((size_t)M*K*sizeof(float)),allocate((size_t)M*K*sizeof(float))};
    const struct ggml_type_traits *traits=ggml_get_type_traits(GGML_TYPE_Q4_K);
    rng=0x6bd03917u^M^(N<<8)^K^salt;ggml_quantize_init(GGML_TYPE_Q4_K);
    for(unsigned input=0;input<2;input++) {
        memset(b[input].data,0xA5,b[input].size);
        for(uint32_t r=0;r<M;r++) {
            for(uint32_t k=0;k<K;k++) row[k]=random_float()*0.25f;
            void *packed=(char*)b[input].data+a_offset+r*row_bytes;
            traits->from_float_ref(row,packed,K);traits->to_float(packed,decoded[input]+(size_t)r*K,K);
            // The production kernels round each dequantized A weight to F16
            // in shared memory, while B and every accumulator remain F32.
            for(uint32_t k=0;k<K;k++) {
                float *v=decoded[input]+(size_t)r*K+k;*v=ggml_fp16_to_fp32(ggml_fp32_to_fp16(*v));
            }
        }
    }
    float *x=b[2].data,*y=b[3].data,*old=allocate(count*sizeof(float));
    for(size_t i=0;i<b[2].size/sizeof(float);i++) x[i]=random_float()*0.1f;
    GemmPush push={M,N,K,stride_b,stride_d,a_offset,b_offset,d_offset};
    for(int candidate=0;candidate<2;candidate++) {
        for(size_t i=0;i<count;i++) y[i]=-999.25f;
        const char *name=full?(candidate?"mul_mm_q4k_gate_up_geglu_intel_full":"mul_mm_q4k_gate_up_geglu_full"):
                              (candidate?"mul_mm_q4k_gate_up_geglu_intel":"mul_mm_q4k_gate_up_geglu");
        dispatch_geglu(name,b,push);
        for(size_t i=0;i<count;i++) {
            const size_t position=i>=d_offset?i-d_offset:0;
            const int written=i>=d_offset && position/stride_d<N && position%stride_d<M;
            if(!written) {if(y[i]!=-999.25f) fail("GEGLU prefix/stride/tail guard changed");continue;}
            const uint32_t r=(uint32_t)(position%stride_d),n=(uint32_t)(position/stride_d);
            if(!isfinite(y[i])) fail("GEGLU output is not finite");
            if(candidate) {if(memcmp(&old[i],&y[i],sizeof(float))) {fprintf(stderr,"GEGLU tiling bits M=%u N=%u K=%u row=%u col=%u\n",M,N,K,r,n);fail("candidate differs from production accumulation");}}
            else {
                double gate=0,up=0;
                for(uint32_t k=0;k<K;k++) {
                    const double value=x[b_offset+(size_t)n*stride_b+k];
                    gate+=decoded[0][(size_t)r*K+k]*value;up+=decoded[1][(size_t)r*K+k]*value;
                }
                double inner=0.7978845608*(gate+0.044715*gate*gate*gate);
                inner=fmin(15.0,fmax(-15.0,inner));
                const double expected=0.5*gate*(1+tanh(inner))*up,error=fabs(y[i]-expected);
                if(error>maximum_absolute_error) maximum_absolute_error=error;
                if(error>0.005+0.002*fabs(expected)) fail("production GEGLU differs from native Q4_K/F16-weight reference");
            }
        }
        if(!candidate) memcpy(old,y,count*sizeof(float));
        cases++;
    }
    pairs++;
    for(unsigned i=0;i<4;i++) free_buffer(b[i]);
    free(row);free(decoded[0]);free(decoded[1]);free(old);
}

int main(int argc,char **argv) {
    (void)argv;if(argc!=1) fail("validator accepts no arguments");init_gpu();
    if(properties.vendorID!=0x8086) fail("this candidate proof requires the Intel worker");
    const uint32_t shapes[][2]={{31,1},{33,15},{67,16},{33,31},{67,32},{33,33},{67,64},{33,80},{67,95},{33,96},{67,97}};
    const uint32_t widths[]={256,5376};const uint32_t salts[]={0u,0x9e3779b9u,0xd1b54a35u};
    for(unsigned salt=0;salt<3;salt++) for(unsigned k=0;k<2;k++) {
        for(unsigned shape=0;shape<sizeof shapes/sizeof shapes[0];shape++) paired_geglu(shapes[shape][0],shapes[shape][1],widths[k],0,salts[salt]);
        for(uint32_t M=32;M<=64;M+=32) for(uint32_t N=32;N<=96;N+=32) paired_geglu(M,N,widths[k],1,salts[salt]);
    }
    check(vkDeviceWaitIdle(device));vkDestroyCommandPool(device,command_pool,NULL);vkDestroyDevice(device,NULL);vkDestroyInstance(instance,NULL);close(gpu_lock);ggml_quantize_free();
    printf("PASS: %u GEGLU native-reference/offset/stride/guard cases\n",cases);
    printf("PASS: %u GEGLU identical-accumulation pairs\n",pairs);
    printf("maximum_absolute_error=%.9g\n",maximum_absolute_error);return 0;
}
