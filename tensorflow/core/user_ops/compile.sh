TF_INC=$(python3 -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB=$(python3 -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
g++ -std=c++11 -shared tensor_generator.cc -o tensor_generator.so -fPIC -I$TF_INC -I$TF_INC/external/nsync/public -L$TF_LIB -D_GLIBCXX_USE_CXX11_ABI=0 -O2 -lprotobuf
