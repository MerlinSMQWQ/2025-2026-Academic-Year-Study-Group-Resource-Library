1. 这是一份基于原版西游记制作的graphRAG项目
2. 可以在.env文件中添加自己的API并在settings.yaml文件中修改对应的API_base
3. 可以将graphml文件导入gephi中，呈现出具体的知识图谱，可以查看knowledgeGraph.png文件，由于graphml文件不包含社区信息，需要在gephi中用leiden algorithm和一些着色手段进行优化
4. 项目具体使用方法请参考：https://microsoft.github.io/graphrag/get_started/