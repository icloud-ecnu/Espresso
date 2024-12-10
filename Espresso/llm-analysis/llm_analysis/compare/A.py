



rank = 0 task name: LoadMicroBatch(buffer_id=0) Begin 1
rank = 0 task name: LoadMicroBatch(buffer_id=0) Finished 1
rank = 0 task name: ForwardPass(buffer_id=0) Begin 2
rank = 1 task name: RecvActivation(buffer_id=0) Begin 1
rank = 2 task name: RecvActivation(buffer_id=0) Begin 1
rank = 0 task name: ForwardPass(buffer_id=0) Finished 2
rank = 0 task name: SendActivation(buffer_id=0) Begin 3
rank = 0 task name: SendActivation(buffer_id=0) Finished 3
rank = 0 task name: LoadMicroBatch(buffer_id=1) Begin 4
rank = 0 task name: LoadMicroBatch(buffer_id=1) Finished 4
rank = 0 task name: ForwardPass(buffer_id=1) Begin 5
rank = 1 task name: RecvActivation(buffer_id=0) Finished 1
rank = 1 task name: ForwardPass(buffer_id=0) Begin 2
rank = 0 task name: ForwardPass(buffer_id=1) Finished 5
rank = 0 task name: SendActivation(buffer_id=1) Begin 6
rank = 1 task name: ForwardPass(buffer_id=0) Finished 2
rank = 1 task name: SendActivation(buffer_id=0) Begin 3
rank = 1 task name: SendActivation(buffer_id=0) Finished 3
rank = 1 task name: RecvActivation(buffer_id=1) Begin 4
rank = 2 task name: RecvActivation(buffer_id=0) Finished 1
rank = 2 task name: LoadMicroBatch(buffer_id=0) Begin 2
rank = 2 task name: LoadMicroBatch(buffer_id=0) Finished 2
rank = 2 task name: ForwardPass(buffer_id=0) Begin 3
rank = 0 task name: SendActivation(buffer_id=1) Finished 6
rank = 0 task name: LoadMicroBatch(buffer_id=2) Begin 7
rank = 0 task name: LoadMicroBatch(buffer_id=2) Finished 7
rank = 0 task name: ForwardPass(buffer_id=2) Begin 8
rank = 1 task name: RecvActivation(buffer_id=1) Finished 4
rank = 1 task name: ForwardPass(buffer_id=1) Begin 5
rank = 1 task name: ForwardPass(buffer_id=1) Finished 5
rank = 1 task name: SendActivation(buffer_id=1) Begin 6
rank = 0 task name: ForwardPass(buffer_id=2) Finished 8
rank = 0 task name: SendActivation(buffer_id=2) Begin 9
rank = 2 task name: ForwardPass(buffer_id=0) Finished 3
rank = 2 task name: BackwardPass(buffer_id=0) Begin 4
rank = 2 task name: BackwardPass(buffer_id=0) Finished 4
rank = 2 task name: SendGrad(buffer_id=0) Begin 5
rank = 2 task name: SendGrad(buffer_id=0) Finished 5
rank = 2 task name: RecvActivation(buffer_id=1) Begin 6
rank = 1 task name: SendActivation(buffer_id=1) Finished 6
rank = 1 task name: RecvActivation(buffer_id=2) Begin 7

rank0 Load,Forward,SencAct,Load,Forward,SencAct............................,load,forward.............,send,......................,load,Forward..................,SendAct...........................
rank1 RecvAct..................,Forward........,SendAct,RecvAct.........................,forward,send.................,RecvAct...,forward...............,SendAct...................................
rank2 RecvAct..................................................,Load,Forward.........................,back,send,RecvAct.......,load.............,Forward................,Backward,SendGrad,RecvAct..


rank = 2 task name: RecvActivation(buffer_id=1) Finished 6
rank = 2 task name: LoadMicroBatch(buffer_id=1) Begin 7
rank = 0 task name: SendActivation(buffer_id=2) Finished 9
rank = 1 task name: RecvActivation(buffer_id=2) Finished 7
rank = 1 task name: ForwardPass(buffer_id=2) Begin 8
rank = 0 task name: LoadMicroBatch(buffer_id=3) Begin 10
rank = 0 task name: LoadMicroBatch(buffer_id=3) Finished 10
rank = 0 task name: ForwardPass(buffer_id=3) Begin 11
rank = 2 task name: LoadMicroBatch(buffer_id=1) Finished 7
rank = 2 task name: ForwardPass(buffer_id=1) Begin 8
rank = 1 task name: ForwardPass(buffer_id=2) Finished 8
rank = 1 task name: SendActivation(buffer_id=2) Begin 9
rank = 0 task name: ForwardPass(buffer_id=3) Finished 11
rank = 0 task name: SendActivation(buffer_id=3) Begin 12
rank = 2 task name: ForwardPass(buffer_id=1) Finished 8
rank = 2 task name: BackwardPass(buffer_id=1) Begin 9
rank = 2 task name: BackwardPass(buffer_id=1) Finished 9
rank = 2 task name: SendGrad(buffer_id=1) Begin 10
rank = 2 task name: SendGrad(buffer_id=1) Finished 10
rank = 2 task name: RecvActivation(buffer_id=2) Begin 11