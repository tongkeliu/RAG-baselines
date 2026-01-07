## 预处理元数据
如果```meta_data['author']['name']```字段为空，则设置为```Unknown```，并增加了```book_summary```字段

分离出的元数据的命名为user_id

## 超参数设置
|超参数名称|参数值|解释|
|---------|------|----|
|chunk_size|500|文本分割块的大小|
|top-k|10|检索保留最相关的文本块个数|
|n_clusters|3|speculative rag聚类数目|
|t|2|astute rag进行自反思的迭代次数|
|数据集分割点|10|在原地对列表进行了分割，前190条为可检索文本，后10条为测试集

## Naive RAG
这个是最原始的rag方法，直接把检索到的文本放在context，然后生成文本

## Dos RAG
这里首先要运行```precreate_nodes.py```来生成可检索文本的节点，然后运行```run_rag.py```来让模型输出最终结果，结果保存在dos_rag/save/answer/{user_id}.json

## Astute RAG
由```naive_rag.py```更改得到，超参数位置一样

## Speculative RAG
由```Astute_RAG.py```更改得到，超参数位置一样

## Dos_Astute_RAG
只对DOS RAG的RAG部分进行了修改，用法与DOS RAG一致