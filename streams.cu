// X  = H , y = W
#include<iostream>
#include<stdio.h>
#include<cuda.h>
#include<ctime>
#include<cstdlib>
#include<cuda_profiler_api.h>
using namespace std;











// serially intializing tensor of the image
void tensor_init(int * image, int N, int H, int W, int C){
    /*
        Initialise the tensor for the convolution operation.
        Runs on the CPU
        N : Batch Size of the image
        H : Height of the image
        W : Width of the image
        C : channels for the kernels
    */
    srand(time(0));
    int tot =  N*H*W*C;
    for(int i = 0; i< tot;i++){
        image[i] = rand()%256; //random initializing of the image tensor// for simulating it as an image
    }
}





//serially intialising the kernel with given dimensions
void kernel_init(int *krnl, int d, int h,  int w,int c){
    /*
        Initialise the kernel(s) for the convolution operation.
        Runs on the CPU
        d : Number of kernel
        h : Height of the kernel
        w : Width of the kernel
        c : channels for the kernels
    */
    int tot = d*h*w*c;
    for(int i = 0; i< tot;i++){
        if(i%2 ==0){
            krnl[i] = rand()%10; 
        }
        else{   
            krnl[i] = -rand()%10;
            //random initializing of the image tensor
        // cout<<krnl[i]<<endl;
        }
    }
}








// intialising the mask for checking sparsity of the block 
void mask_init(int *mask,int N,int H,int W,int sparsity_perc){
    /*
        Initialise the tensor for the convolution operation.
        Runs on the CPU
        N : Batch Size of the image
        H : Height of the image
        W : Width of the image
    */
    int tot = N*H*W;
    for(int i = 0; i< tot;i++){
        if(rand()%100<=sparsity_perc){
            mask[i] = 0;
        }
        else{
            mask[i] = 1;
        }  //random initializing of the image tensor
    // cout<<mask[i]<<endl;
    }
    
}




// ************************ device kernels **************** to be optimizzed ***************************************
__device__ bool checksparse(int *d_mask,int cx,int cy,int H, int W, int C,int h,int w,int S,int n){// may be i can have some more conditions
    /*
        device function to check for sparsity

        (device int *) d_mask : pointer to the mask of the image
        (int) n: number of the image
        (int) h: height of the kernels
        (int) w: Weight of the kernels
        (int) c_x: x coordinate of the center
        (int) c_y: y coordinate of the center
    
    */
    int x = 0;
    int y = 0;
    for( int l=-(h-1)/2; l <= (h-1)/2; l++ ){
        for( int p=-(w-1)/2; p <= (w-1)/2; p++ ){
            x = cx + l;
            y = cy + p;
            if( d_mask[n*H*W  + W*y  +  x ] == 1 ){
                return false;
            }
        }
    }
    return true;
}






__global__ void gather(int *d_mask, int *d_tensor, int *d_mat,unsigned int *row_address, int * d_row_map, int N , int H , int W , int h, int w, int C , int S ){ 
    /*
        Gather kernel from the paper to check for sparse and non sparse parts of image for convolution

        (device int *) d_mask : pointer to the mask of the image
        (device int *) d_tensor :  pointer to the tensor containing the all the images
        (device int *) d_mat : pointer with memmory alloc to store every non sparse part of thhe images
        (device int *) row_address : pointer to single integer containing the number of non sparse part of the image
        (int) N: number of the images in the given tensor
        (int) H: Height of the image
        (int) W: Weight of the image
        (int) C: Channels of the image
        (int) h: height of the kernels
        (int) w: Weight of the kernels
        
    */



    int id2 = blockIdx.x*blockDim.x + threadIdx.x;
    int in = blockIdx.y;
    int x_dim = id2%W;// along the height of the image
    int y_dim = id2/W;// along the length oh the image
    if(x_dim > 0 && x_dim/S + h < H/S){// condition considering s = 1 for now
        if(y_dim > 0 && y_dim/S +w < W/S){
            int cen_x =  x_dim + (h-1)/2;
            int cen_y =  y_dim + (w-1)/2;
            // printf("%d,%d,%d\n",checksparse(d_mask,x_dim,y_dim,H,W,C,h,w,S,in),cen_x,cen_y); 
            if(!checksparse(d_mask,x_dim,y_dim,H,W,C,h,w,S,in)){
                unsigned   int val = atomicInc(row_address,1000000);
                int col_index = 0;
                for( int l=-(h-1)/2; l <= (h-1)/2; l++ ){
                    for( int p=-(w-1)/2; p <= (w-1)/2; p++ ){
                        for( int q=0; q < C; q++){
                            d_mat[val*h*w*C+col_index] = d_mask[in*(H/S)*(W/S)+((int)((cen_x+l)/S))*(W/S)+((int)((cen_y+p)/S))]?d_tensor[in*H*W*C+(cen_x+l)*W*C+(cen_y+p)*C+q]:0;
                            col_index += 1;
                        }
                    }
                }
                d_row_map[val*3+0] = x_dim; /* Store the original x-coordinate corresponding to a row into a map */
                d_row_map[val*3+1] = y_dim; /* Store the original y-coordinate corresponding to a row into a map */
                d_row_map[val*3+2] = in; /* Store the image corresponding to a row in a map */
                // printf("%d\n",val);
            }
        }
    }
}










__global__ void convolution(int *d_mat,int *d_kernel,unsigned int *number_rows ,int d,int *output_mat,int h,int w,int C){
    /*
        The most basic implementation of the cuda kernel;
        (int *)d_mat : pointer to the conovoluted results for all the non scarse part of the original image
        (int *)d_kernel  : kernel for the coonvoltion(d kernels)
        (int *)output_mat :  pointer for finally storing the output of the matrix
        (unsigned int):  int containing the number of non sparse convolution block
        (int) N: number of the images in the given tensor
        (int) H: Height of the image
        (int) W: Weight of the image
        (int) C: Channels of the image
        (int) h: height of the kernels
        (int) w: Weight of the kernels
        (int) d : number of kernels
    
    */
    int t_idx = blockDim.x*blockIdx.x + threadIdx.x;// for the number of the element being changed
    int t_idy = blockDim.y*blockIdx.y + threadIdx.y;// for the number of kernels
    output_mat[t_idx*d + t_idy] = 0;
    int offset = h*w*C;
    if(t_idx < *number_rows && t_idy < d){
        // now the convolution part
        for(int i = 0; i < h*w*C; i++ ){
            output_mat[t_idx*d + t_idy] += d_kernel[t_idy*h*w  + i]*d_mat[offset*t_idx + i]; 
            // printf("%d,%d,\n",d_kernel[t_idy*d +i],d_mat[offset*t_idx + i]);
        }
        // printf("%d,%d,%d\n",t_idx,t_idy,output_mat[t_idx*d + t_idy]);
    }
    
}







__global__ void scatter(int *output_mat, int *d_row_map, unsigned int *number_rows, int *output,int H,int W,int d,int h,int w){
    /*
        Putting the peices back together in the final image(restoring the final output part of the kernel
        (int *)output_mat : pointer to the conovoluted results for all the non scarse part of the original image
        (int *)d_row_map  : pointer to the center positions non sparse part of the image
        (int *)output :  pointer to the final image after convolutions
        (int) N: number of the images in the given tensor
        (int) H: Height of the image
        (int) W: Weight of the image
        (int) C: Channels of the image
        (int) h: height of the kernels
        (int) w: Weight of the kernels
        (int) d : number of kernels
    
    */
    int image_size = (H - h + 1)*(W-w+1);
    // image size after the convolution happens
    int t_idx = blockIdx.x*blockDim.x + threadIdx.x;// The number of convs in the output matrux
    int t_idy = blockDim.y*blockIdx.y + threadIdx.y;// The number of output kernels
    // printf("%d,%d,%d \n",t_idx,t_idy, 0);
    if(t_idx<*number_rows && t_idy <d){
        int c_x = d_row_map[t_idx*3] - (h-1)/2; // convert the center to convoluted positions
        int c_y = d_row_map[t_idx*3 + 1] - (w-1)/2;
        int N = d_row_map[t_idx*3 + 2];
        output[N*(image_size*d) + t_idy*(image_size) + W*(c_y) + c_x] =  output_mat[t_idx*d + t_idy ];
        //printf("%d,%d,%d\n",output[N*(image_size*d) + t_idy*(image_size) + W*(c_y) + c_x],output_mat[t_idx*d + t_idy ],N*(image_size*d) + t_idy*(image_size) + W*(c_y) + c_x);
    }

}












int main(){
    // taking input of the image(tensor) dimnsions
    int BLOCK_SIZE = 32;
    int N,H,W,C;
    /*
        (int) N: number of the images in the given tensor
        (int) H: Height of the image
        (int) W: Weight of the image
        (int) C: Channels of the image
    */
    cout<<"Gimme Image Block Dimensions"<<endl;
    N = 4;
    H = 256;
    W = 256;
    C  = 3;
    
    int *tensor = (int *)malloc(N*H*W*C*sizeof(int));
    
    tensor_init(tensor,N,H,W,C);
    
    int h,w,d;
    /*
    
        (int) h: height of the kernels
        (int) w: Weight of the kernels
        (int) d : number of kernels

    */
    int c = C; 
    cout<<"Gimme krnl Block Dimension"<<endl;
    d = 16;
    h = 4;
    w = 4;

    int *kernel = (int *)malloc(sizeof(int)*h*w*c*d);
    kernel_init(kernel,d,h,w,C);
    // space for d kernels

    int per_sp;

    cout<<"Gimme Percent Sparcity of the block"<<endl;

    per_sp =70;

    int S = 1;// assuming the mask dimension to be 1 for now
    int *mask = (int * )malloc(sizeof(int)*N*H*W*C);
    mask_init(mask,N,H,W,per_sp);

    int num_images = 2;
    int n_streams = N/2;


    // memory allocation for tensor kernel and the mask on the device
    int *d_tensor; 
    int *d_kernel; 
    int *d_mask;
    cudaMalloc(&d_tensor,sizeof(int)*N*H*W*C);// 4-D tensor containing images for the convolution operation
    cudaMalloc(&d_kernel,sizeof(int)*d*h*w*c);// for the kernels to stored in the matrix
    cudaMalloc(&d_mask,sizeof(int)*N*H*W); //mask for checking the sparsity of blocks for the kernel 

    // memory copying to the device
    cudaMemcpy( d_kernel, kernel, sizeof(int)*d*h*w*c, cudaMemcpyHostToDevice );
    cudaMemcpy( d_mask, mask, sizeof(int)*N*H*W, cudaMemcpyHostToDevice );
    cudaMemcpy( d_tensor, tensor, sizeof(int)*N*H*W*C, cudaMemcpyHostToDevice );



    // gatther kernel to fill on the device
    int * d_mat;//
    int * d_row_map; 
    unsigned int *row_address;  
    
    cudaMalloc(&d_mat,sizeof(int)*h*w*C*(H-h+1)*(W-w+1)*N); //  considering that all the parts will be in the imsge
    cudaMalloc(&row_address,n_streams*sizeof( unsigned int));
    
    cudaMemset(&row_address, 0, n_streams*sizeof(unsigned int) );
    cudaMalloc(&d_row_map,sizeof(int)*(H-h+1)*(W-w+1)*N*3);
    // create streams:
    // it can roughly handle about 1000 images at once
    cudaStream_t streams[1000]; /* Declaring a set of CUDA streams */
    for( int i=0; i<n_streams; i++ ) cudaStreamCreate(&streams[i]); /* Initializing a set of streams to work on a set of each image */



    // creating memory for the intermediate kernels
    int * output_mat; /// for the putput of the gather kernel
    cudaMalloc(&output_mat,sizeof(int)*(H-h+1)*(W-w+1)*d*N); 


    // final output matrix we all know its dimensionb already
    int * output;
    cudaMalloc(&output,sizeof(int)*N*(H-h+1)*(W-w+1)*d);




    // profiling features          -----------------------
    cudaEvent_t     start,stop; /* CUDA events to time the program */

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    

    // blocks and threads for diffrent kernel plaunches -----------


    // for the gather kernle
    dim3 Block(H*W/BLOCK_SIZE,num_images,1);
    dim3 Thread(BLOCK_SIZE,1,1);
    
    //convolution kernel
    dim3 Block_c(((H-h+1)*(W-w+1)*num_images)/BLOCK_SIZE,d,1);
    
    dim3 Thread_c(BLOCK_SIZE,1,1);




    // offset for diffrent arrays -------------
    int offset;   // tensor fooffset
    int mask_offset; // mask offset
    int mat_offset; // d_mat offsaet
    int map_offset; // d_row_map offset
    int o_offset; //output_mat offset
    int om_offset; // output offset
    unsigned int *number_rows = (unsigned int *)malloc(sizeof(int)*n_streams );


    // Allocating memory for the output tensor
    int * h_output = (int *)malloc(sizeof(int)*N*(H-h+1)*(W-w+1)*d);


    //scatter kernel --- 
    
    
    dim3 Block_s(((H-h+1)*(W-w+1)*num_images)/BLOCK_SIZE,d,1);
    dim3 Thread_s(BLOCK_SIZE,1,1);

    //lanching the diffrent streams
    for(int j=0; j<n_streams; j++){

        /* Initialize a set of off-sets for each stream */
        offset = j*H*W*C*num_images; // tensor offset will be needed
        mask_offset = j*(H)*(W)*num_images; /// mask offset will be needed
        mat_offset = h*w*C*(H-h+1)*(W-w+1)*j*num_images;//matrix offset for the value to be in the matrix
        map_offset = 3*(H-h+1)*(W-w+1)*j*num_images;//offset for d_row_map
        o_offset = (H-h+1)*(W-w+1)*d*j*num_images;//offset for convolution output
        om_offset = d*(H-h+1)*(W-w+1)*j*num_images;//final output offset


        // now the kernels.............. 
        // gether kernel
        gather<<<Block, Thread, 0, streams[j]>>>(&d_mask[mask_offset], &d_tensor[offset], &d_mat[mat_offset],&row_address[j], &d_row_map[map_offset],N , H , W , h, w, C , S);
        // cudaMemcpyAsync(&number_rows[j], &row_address[j], sizeof(unsigned int), cudaMemcpyDeviceToHost,streams[j]);
        //convolution kernel
        convolution<<<Block_c, Thread_c,0, streams[j]>>>(&d_mat[mat_offset], d_kernel, &row_address[j], d, &output_mat[om_offset],h,w,C);
        // cout<<"kernel went through"<<endl;

        // convert the kernel back to its original form
        scatter<<<Block_s,Thread_s, 0 , streams[j]>>>(&output_mat[om_offset], &d_row_map[map_offset], &row_address[j], &output[o_offset],H, W, d, h, w);
        
        
    }
    cudaMemcpy(h_output,output,sizeof(int)*(H-h+1)*(W-w+1)*d*N,cudaMemcpyDeviceToHost);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float   run_time = 0.0;
    cudaEventElapsedTime(&run_time,start,stop);
    cout<<run_time<<endl;
    
    
    // for(int k = 0;k<N;k++){
    //     for(int p = 0; p<d;p++){
    //         cout<<"image"<<" "<<k<<" "<<"kernel"<<" "<<p<<endl;
    //         for(int i = 0; i<(H-h+1);i++){
    //             for(int j = 0; j<(W-w+1);j++){
    //                 cout<<h_output[k*(H-h+1)*(W-w+1)*d + p*(H-h+1)*(W-w+1) + i*(W-w+1)+ j ]<<" ";
    //             }
    //             cout<<endl;
    //         }    
    //         cout<<endl;
    //     }
    //     cout<<endl;
    // }



    // Destroying all the streams in rthe 
    for( int i=0; i<n_streams; i++ ) cudaStreamDestroy(streams[i]);
    return 0;

}