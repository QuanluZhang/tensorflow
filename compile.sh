rm tensorflow/core/user_ops/tensor_generator.so
mv tensorflow/core/user_ops/tensor_generator.cc ~/

bazel build --config=opt --config=cuda --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" //tensorflow/tools/pip_package:build_pip_package
bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
pip3 uninstall tensorflow
pip3 install /tmp/tensorflow_pkg/tensorflow-1.3.1-cp35-cp35m-linux_x86_64.whl --user

mv ~/tensor_generator.cc tensorflow/core/user_ops/
cd tensorflow/core/user_ops
./compile.sh
