## faster_rcnn模型转化

1. 查看输入输出节点

   使用网络训练后会生成3个ckpt加一个checkpoint文件：

   ```
   checkpoint
   
   voc_8001model.ckpt.data-00000-of-00001
   
   voc_8001model.ckpt.index
   
   voc_8001model.ckpt.meta
   ```

   输入和输出tensor的确定是ckpt模型转pb模型的关键，采用Netron软件可以直接打开`.meta`文件，查看网络结构图得到模型的输入输出节点名称。也可以通过如下代码通过ckpt文件直接生成输入输出节点名称。

   ```
   import tensorflow as tf
   ckpt = 'F:/gree_test/tensorflow/voc_8001model.ckpt'
   b=open('a.txt',mode='a+')
   with tf.Session() as sess:
       saver = tf.train.import_meta_graph(ckpt + '.meta', clear_devices=True)
       graph_def = tf.get_default_graph().as_graph_def(add_shapes=True)
       node_list = [n.name for n in graph_def.node]
       for node in node_list:
           b.write(str(node)+"\n")
           print("node_name", node)
   ```

   因为faster_rcnn模型比较大，打印出来的节点名称非常多，需要熟悉网络结构的输入输出。最方便的就是直接在网络测试代码中寻找网络节点，在原始文件lib/nets/network.py找到如下代码：

       def test_image(self, sess, image, im_info):
           feed_dict = {self._image: image,
                        self._im_info: im_info}
       cls_score, cls_prob, bbox_pred, rois = sess.run([self._predictions["cls_score"],
                                                        self._predictions['cls_prob'],
                                                        self._predictions['bbox_pred'],
                                                        self._predictions['rois']],
                                                       feed_dict=feed_dict)
       return cls_score, cls_prob, bbox_pred, rois

   这里就是tensorflow开始run模型，输入的参数就是我们想要的，根据run()的调用方式，前面的列表是输出的tensor，后面的feed_dict字典是喂给图的数据和输入的tensor，这里可以提取打印一下这几个参数，最终得到我们的输入和输出节点。输入节点：Placeholder：0，Placeholder_1：0，输出节点：vgg_16_3/cls_score/BiasAdd:0, vgg_16_3/cls_prob:0, add:0, vgg_16_1/rois/concat:0.

2. 固化pb模型

   将CKPT 转换成 PB格式的文件的过程可简述如下：

   - 通过传入 CKPT 模型的路径得到模型的图和变量数据

   - 通过 import_meta_graph 导入模型中的图

   - 通过 saver.restore 从模型中恢复图中各个变量的数据

   - 通过 graph_util.convert_variables_to_constants 将模型持久化

     注意改变输出节点，转化代码如下：

   ```
   import tensorflow as tf
   from tensorflow.python.framework import graph_util
   from tensorflow.python.platform import gfile
   def freeze_graph(ckpt, output_graph):
       output_node_names = 'vgg_16_3/cls_score/BiasAdd:0,vgg_16_3/cls_prob:0,add:0,vgg_16_1/rois/concat:0'
       saver = tf.compat.v1.train.import_meta_graph(ckpt + '.meta', clear_devices=True)
       graph = tf.get_default_graph()
       input_graph_def = graph.as_graph_def()
       with tf.Session() as sess:
           saver.restore(sess, ckpt)
           output_graph_def = graph_util.convert_variables_to_constants(
               sess=sess,
               input_graph_def=input_graph_def,
               output_node_names=output_node_names.split(',')
           )
           with tf.gfile.GFile(output_graph, 'wb') as fw:
               fw.write(output_graph_def.SerializeToString())
           print('{} ops in the final graph.'.format(len(output_graph_def.node)))
   ckpt = 'F:/gree_test/tensorflow/voc_8001model.ckpt'
   pb = 'F:/gree_test/tensorflow/bert_model.pb'
   if __name__ == '__main__':
       freeze_graph(ckpt, pb)
   ```

3. ONNX模型转化

   安装tensorflow转换onnx模型的转化工具

   `pip install tf2onnx`

   运行下列命令将pb模型转化为onnx模型

   ```bash
   python -m tf2onnx.convert\
       --input bert_model.pb\
       --inputs Placeholder：0,Placeholder_1：0\
       --outputs vgg_16_3/cls_score/BiasAdd:0,vgg_16_3/cls_prob:0,add:0,vgg_16_1/rois/concat:0.\
       --output model.onnx\
       --opset10
   ```

   

4. 存在问题

   （1）在利用tf2onnx转化onnx模型时，在过程中会出现以下错误

   `ERROR - Tensorflow op [vgg_16_1/rois/non_max_suppression/NonMaxSuppressionV3: NonMaxSuppressionV3] is not supported`

   由于有不支持的op，需要在转换代码中加上--continue_on_error，成功转成onnx。

   （2）在利用转成的onnx模型进行tensorrt测试时，出现了如下错误：

   `ERROR - Failed to convert node vgg_16_1/pool5/crops`
   `OP=CropAndResize`

   这里使用ONNX-graphsurgeon来修改onnx model

   将vgg_16_1/pool5/crops这个节点的名称修改为"CropAndResize"

   （3）运行修改节点后的onnx模型会出现如下错误：

   `[E] [TRT] (Unnamed Layer* 154) [Constant]: invalid weights type of Bool`

   



