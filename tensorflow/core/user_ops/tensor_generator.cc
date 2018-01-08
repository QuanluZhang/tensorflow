#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/framework/register_types.h"

//#include "auxiliary_header.h"

using namespace tensorflow;


REGISTER_OP("TensorGeneratorTmp")
    .Attr("T: type")
    .Input("tensor_shape: int32")
    .Output("out_tensor: T") // what if data type is different
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        //c->set_output(0, c->input(0));
        return Status::OK();
    });
    //.Doc(R"doc(Generate different type/shape of tensor according to input data)doc");

template <typename T>
class TensorGeneratorTmpOp : public OpKernel {
  public:
    explicit TensorGeneratorTmpOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* context) override {
        const Tensor& input_tensor = context->input(0);
        auto input = input_tensor.flat<int32>();
        CHECK_GE(input.size(), 1) << "input size " << input.size()
                                  << " should be larger than or equal to 2";

        DataType out_data_type;
        int type_num = input(0);
        if (type_num >= 1 && type_num <= 20) {
            out_data_type = (DataType)type_num;
        }
        else {
            CHECK_GE(1, 2) << "TensorGenerator: invalid data type " << type_num
                           << " end";
        }

        Tensor* output_tensor = NULL;
        TensorShape new_shape = TensorShape();
        for (int i = 1; i < input.size(); i++) {
            new_shape.AddDim(input(i));
        }
        //new_shape.set_data_type_pub(DT_FLOAT);
        new_shape.set_data_type_pub(out_data_type);
        OP_REQUIRES_OK(context, context->allocate_output(0, new_shape, &output_tensor));
        //auto output_flat = output_tensor->flat<float>();

        //const int N = input.size();
        //const int N = output_flat.size();
        //printf("XXXXXXXXXXXXXXXXXXXXXXXXX%d\n", N);
        //for (int i = 0; i < N; i++) {
        //    output_flat(i) = 1;
        //}

        //if (N > 0) output_flat(0) = input(0);
    }

    //bool IsExpensive() override { return false; }
};

//REGISTER_KERNEL_BUILDER(Name("TensorGenerator").Device(DEVICE_CPU), TensorGeneratorOp);
#define REGISTER_KERNEL(type)                                                   \
    REGISTER_KERNEL_BUILDER(                                                    \
        Name("TensorGeneratorTmp").Device(DEVICE_CPU).TypeConstraint<type>("T"),   \
        TensorGeneratorTmpOp<type>)

TF_CALL_ALL_TYPES(REGISTER_KERNEL);

#undef REGISTER_KERNEL
