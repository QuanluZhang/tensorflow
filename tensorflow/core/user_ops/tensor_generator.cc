#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/framework/register_types.h"

//#include "auxiliary_header.h"

using namespace tensorflow;


REGISTER_OP("TensorGenerator")
    .Attr("T: type")
    .Input("tensor_shape: int32")
    .Output("out_tensor: T") // what if data type is different
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        //c->set_output(0, c->input(0));
        return Status::OK();
    });
    //.Doc(R"doc(Generate different type/shape of tensor according to input data)doc");

template <typename T>
class TensorGeneratorOp : public OpKernel {
  public:
    explicit TensorGeneratorOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* context) override {
        const Tensor& input_tensor = context->input(0);
        auto input = input_tensor.flat<int32>();

        Tensor* output_tensor = NULL;
        TensorShape new_shape = TensorShape();
        new_shape.AddDim(2);
        new_shape.AddDim(2);
        new_shape.AddDim(2);
        new_shape.set_data_type_pub(DT_FLOAT);
        OP_REQUIRES_OK(context, context->allocate_output(0, new_shape, &output_tensor));
        auto output_flat = output_tensor->flat<float>();

        //const int N = input.size();
        const int N = output_flat.size();
        printf("XXXXXXXXXXXXXXXXXXXXXXXXX%d\n", N);
        for (int i = 0; i < N; i++) {
            output_flat(i) = 1;
        }

        //if (N > 0) output_flat(0) = input(0);
    }
};

//REGISTER_KERNEL_BUILDER(Name("TensorGenerator").Device(DEVICE_CPU), TensorGeneratorOp);
#define REGISTER_KERNEL(type)                                                   \
    REGISTER_KERNEL_BUILDER(                                                    \
        Name("TensorGenerator").Device(DEVICE_CPU).TypeConstraint<type>("T"),   \
        TensorGeneratorOp<type>)

TF_CALL_ALL_TYPES(REGISTER_KERNEL);

#undef REGISTER_KERNEL
