#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/framework/register_types.h"
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

//#include "auxiliary_header.h"
#define TENSOR_CONTENT_FILE "/home/quzha/static_analysis/result/dump_tensor_content.txt"

using namespace tensorflow;


REGISTER_OP("TensorGeneratorTma")
    .Attr("T: type")
    .Attr("tensor_name: string = 'null'")
    .Input("tensor_shape: int32")
    .Output("out_tensor: T") // what if data type is different
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        //c->set_output(0, c->input(0));
        return Status::OK();
    });
    //.Doc(R"doc(Generate different type/shape of tensor according to input data)doc");

template <typename T>
class TensorGeneratorTmaOp : public OpKernel {
  public:
    explicit TensorGeneratorTmaOp(OpKernelConstruction* context) : OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("tensor_name", &tensor_name_));
    }

    void Compute(OpKernelContext* context) override {
        static std::map<std::string, char*> map_of_tensors;
        if (map_of_tensors.size() == 0) {
            int fd = open(TENSOR_CONTENT_FILE, O_RDONLY);
            while (1) {
                int src_id, output_index;
                int ret = read(fd, &src_id, sizeof(src_id));
                if (ret == 0) break;
                if (ret != sizeof(src_id)) { printf("src_id error\n"); exit(-1); }
                ret = read(fd, &output_index, sizeof(output_index));
                if (ret != sizeof(output_index)) { printf("output_index error\n"); exit(-1); }

                size_t name_len;
                ret = read(fd, &name_len, sizeof(name_len));
                if (ret != sizeof(name_len)) { printf("name_len error\n"); exit(-1); }
                char src_op_name[1024];
                ret = read(fd, src_op_name, name_len);
                if (ret != name_len) { printf("name error\n"); exit(-1); }
                src_op_name[name_len] = '\0';

                /*size_t proto_size;
                ret = read(fd, &proto_size, sizeof(proto_size));
                if (ret != sizeof(proto_size)) { printf("proto_size error\n"); exit(-1); }
                TensorProto* tmp_proto = new TensorProto();
                tmp_proto->ParseFromFileDescriptor(fd);
                printf("parsefromfiledescriptor\n");
                map_of_tensors.emplace(src_op_name, tmp_proto);*/

                size_t buf_size;
                ret = read(fd, &buf_size, sizeof(buf_size));
                if (ret != sizeof(buf_size)) { printf("buf_size error\n"); exit(-1); }
                char* tmp_buf = new char[buf_size];
                ret = read(fd, tmp_buf, buf_size);
                if (ret != buf_size) { printf("buf error\n"); exit(-1); }

                map_of_tensors.emplace(src_op_name, tmp_buf);
            }
            close(fd);
        }

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
        printf("name: %s\n", tensor_name_.data());
        new_shape.set_data_type_pub(out_data_type);
        OP_REQUIRES_OK(context, context->allocate_output(0, new_shape, &output_tensor));
        if (map_of_tensors.find(tensor_name_) != map_of_tensors.end()) {
            if (!output_tensor->FromProto(*map_of_tensors[tensor_name_])) {
                printf("Failed to parse TensorProto\n");
                exit(-1);
            }
        }
        else {
            printf("there is no tensor content for this output_tensor");
        }
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
  private:
    std::string tensor_name_;
};

//REGISTER_KERNEL_BUILDER(Name("TensorGenerator").Device(DEVICE_CPU), TensorGeneratorOp);
#define REGISTER_KERNEL(type)                                                   \
    REGISTER_KERNEL_BUILDER(                                                    \
        Name("TensorGeneratorTma").Device(DEVICE_CPU).TypeConstraint<type>("T"),   \
        TensorGeneratorTmaOp<type>)

TF_CALL_ALL_TYPES(REGISTER_KERNEL);

#undef REGISTER_KERNEL
