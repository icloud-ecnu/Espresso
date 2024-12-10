rank = 0 task name: LoadMicroBatch(buffer_id=0) Begin 1
rank = 0 task name: LoadMicroBatch(buffer_id=0) Finished 1
rank = 0 task name: ForwardPass(buffer_id=0) Begin 2
[['F', 'F', 'F', 'F', 'B', 'B', 'B', 'B'], ['F', 'F', 'F', 'F', 'B', 'B', 'B', 'B'], ['F', 'F', 'F', 'F', 'B', 'B', 'B', 'B'], ['F', 'F', 'F', 'F', 'B', 'B', 'B', 'B']]
None
rank = 3 task name: RecvActivation(buffer_id=0) Begin 1
_exec_recv_activations self.prev_stage = 2 rank = 3
[['F', 'F', 'F', 'F', 'B', 'B', 'B', 'B'], ['F', 'F', 'F', 'F', 'B', 'B', 'B', 'B'], ['F', 'F', 'F', 'F', 'B', 'B', 'B', 'B'], ['F', 'F', 'F', 'F', 'B', 'B', 'B', 'B']]
None
rank = 2 task name: RecvActivation(buffer_id=0) Begin 1
_exec_recv_activations self.prev_stage = 1 rank = 2
[['F', 'F', 'F', 'F', 'B', 'B', 'B', 'B'], ['F', 'F', 'F', 'F', 'B', 'B', 'B', 'B'], ['F', 'F', 'F', 'F', 'B', 'B', 'B', 'B'], ['F', 'F', 'F', 'F', 'B', 'B', 'B', 'B']]
None
rank = 1 task name: RecvActivation(buffer_id=0) Begin 1
_exec_recv_activations self.prev_stage = 0 rank = 1
rank = 0 task name: ForwardPass(buffer_id=0) Finished 2
rank = 0 task name: SendActivation(buffer_id=0) Begin 3
rank = 0 task name: SendActivation(buffer_id=0) Finished 3
rank = 0 task name: LoadMicroBatch(buffer_id=1) Begin 4
rank = 0 task name: LoadMicroBatch(buffer_id=1) Finished 4
rank = 0 task name: ForwardPass(buffer_id=1) Begin 5
rank = 1 task name: RecvActivation(buffer_id=0) Finished 1
rank = 1 task name: ForwardPass(buffer_id=0) Begin 2
/home/ecnu/iSomer/llm/collie/collie/utils/pipeline_engine.py:953: UserWarning: The .grad attribute of a Tensor that is not a leaf Tensor is being accessed. Its .grad attribute won't be populated during autograd.backward(). If you indeed want the .grad field to be populated for a non-leaf Tensor, use .retain_grad() on the non-leaf Tensor. If you access the non-leaf Tensor by mistake, make sure you access the leaf Tensor instead. See github.com/pytorch/pytorch/pull/30531 for more informations. (Triggered internally at aten/src/ATen/core/TensorBody.h:486.)
  if t.grad is not None:
rank = 0 task name: ForwardPass(buffer_id=1) Finished 5
rank = 0 task name: SendActivation(buffer_id=1) Begin 6
rank = 1 task name: ForwardPass(buffer_id=0) Finished 2
rank = 1 task name: SendActivation(buffer_id=0) Begin 3
rank = 1 task name: SendActivation(buffer_id=0) Finished 3
rank = 1 task name: RecvActivation(buffer_id=1) Begin 4
_exec_recv_activations self.prev_stage = 0 rank = 1
rank = 2 task name: RecvActivation(buffer_id=0) Finished 1
rank = 2 task name: ForwardPass(buffer_id=0) Begin 2
/home/ecnu/iSomer/llm/collie/collie/utils/pipeline_engine.py:953: UserWarning: The .grad attribute of a Tensor that is not a leaf Tensor is being accessed. Its .grad attribute won't be populated during autograd.backward(). If you indeed want the .grad field to be populated for a non-leaf Tensor, use .retain_grad() on the non-leaf Tensor. If you access the non-leaf Tensor by mistake, make sure you access the leaf Tensor instead. See github.com/pytorch/pytorch/pull/30531 for more informations. (Triggered internally at aten/src/ATen/core/TensorBody.h:486.)
  if t.grad is not None:
rank = 1 task name: RecvActivation(buffer_id=1) Finished 4
rank = 1 task name: ForwardPass(buffer_id=1) Begin 5
/home/ecnu/iSomer/llm/collie/collie/utils/pipeline_engine.py:953: UserWarning: The .grad attribute of a Tensor that is not a leaf Tensor is being accessed. Its .grad attribute won't be populated during autograd.backward(). If you indeed want the .grad field to be populated for a non-leaf Tensor, use .retain_grad() on the non-leaf Tensor. If you access the non-leaf Tensor by mistake, make sure you access the leaf Tensor instead. See github.com/pytorch/pytorch/pull/30531 for more informations. (Triggered internally at aten/src/ATen/core/TensorBody.h:486.)
  if t.grad is not None:
rank = 0 task name: SendActivation(buffer_id=1) Finished 6
rank = 0 task name: LoadMicroBatch(buffer_id=2) Begin 7
rank = 0 task name: LoadMicroBatch(buffer_id=2) Finished 7
rank = 0 task name: ForwardPass(buffer_id=2) Begin 8
rank = 0 task name: ForwardPass(buffer_id=2) Finished 8
rank = 0 task name: SendActivation(buffer_id=2) Begin 9
rank = 1 task name: ForwardPass(buffer_id=1) Finished 5
rank = 1 task name: SendActivation(buffer_id=1) Begin 6
rank = 2 task name: ForwardPass(buffer_id=0) Finished 2
rank = 2 task name: SendActivation(buffer_id=0) Begin 3
rank = 3 task name: RecvActivation(buffer_id=0) Finished 1
rank = 3 task name: LoadMicroBatch(buffer_id=0) Begin 2
rank = 2 task name: SendActivation(buffer_id=0) Finished 3
rank = 2 task name: RecvActivation(buffer_id=1) Begin 4
rank = 3 task name: LoadMicroBatch(buffer_id=0) Finished 2
rank = 3 task name: ForwardPass(buffer_id=0) Begin 3
_exec_recv_activations self.prev_stage = 1 rank = 2
/home/ecnu/iSomer/llm/collie/collie/utils/pipeline_engine.py:953: UserWarning: The .grad attribute of a Tensor that is not a leaf Tensor is being accessed. Its .grad attribute won't be populated during autograd.backward(). If you indeed want the .grad field to be populated for a non-leaf Tensor, use .retain_grad() on the non-leaf Tensor. If you access the non-leaf Tensor by mistake, make sure you access the leaf Tensor instead. See github.com/pytorch/pytorch/pull/30531 for more informations. (Triggered internally at aten/src/ATen/core/TensorBody.h:486.)
  if t.grad is not None:
rank = 1 task name: SendActivation(buffer_id=1) Finished 6
rank = 1 task name: RecvActivation(buffer_id=2) Begin 7
_exec_recv_activations self.prev_stage = 0 rank = 1
rank = 2 task name: RecvActivation(buffer_id=1) Finished 4
rank = 2 task name: ForwardPass(buffer_id=1) Begin 5
/home/ecnu/iSomer/llm/collie/collie/utils/pipeline_engine.py:953: UserWarning: The .grad attribute of a Tensor that is not a leaf Tensor is being accessed. Its .grad attribute won't be populated during autograd.backward(). If you indeed want the .grad field to be populated for a non-leaf Tensor, use .retain_grad() on the non-leaf Tensor. If you access the non-leaf Tensor by mistake, make sure you access the leaf Tensor instead. See github.com/pytorch/pytorch/pull/30531 for more informations. (Triggered internally at aten/src/ATen/core/TensorBody.h:486.)
  if t.grad is not None:
rank = 1 task name: RecvActivation(buffer_id=2) Finished 7
rank = 1 task name: ForwardPass(buffer_id=2) Begin 8
rank = 0 task name: SendActivation(buffer_id=2) Finished 9
rank = 0 task name: LoadMicroBatch(buffer_id=3) Begin 10
rank = 0 task name: LoadMicroBatch(buffer_id=3) Finished 10
rank = 0 task name: ForwardPass(buffer_id=3) Begin 11
rank = 0 task name: ForwardPass(buffer_id=3) Finished 11
rank = 0 task name: RecvGrad(buffer_id=0) Begin 12
_exec_recv_grads self.prev_stage = -1 rank = 0
rank = 2 task name: ForwardPass(buffer_id=1) Finished 5
rank = 2 task name: SendActivation(buffer_id=1) Begin 6
rank = 1 task name: ForwardPass(buffer_id=2) Finished 8
rank = 1 task name: RecvGrad(buffer_id=0) Begin 9
_exec_recv_grads self.prev_stage = 0 rank = 1
rank = 3 task name: ForwardPass(buffer_id=0) Finished 3
rank = 3 task name: BackwardPass(buffer_id=0) Begin 4
rank = 3 task name: BackwardPass(buffer_id=0) Finished 4
rank = 3 task name: SendGrad(buffer_id=0) Begin 5
_exec_send_grads self.prev_stage = 2 rank = 3