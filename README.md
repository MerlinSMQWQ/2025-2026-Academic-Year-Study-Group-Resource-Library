# 2025-2026-Academic-Year-Study-Group-Resource-Library
2025-2026学年前沿学习小组资料库

这个资料库用于存放小组内分享的资料，目前设置为public，即使是非小组内的成员也可以浏览或者贡献，欢迎你的贡献，让我们的资料越来越丰富！！！

# 关于commit message的一些写法说明
commit message要尽量做到清晰，一个清晰的commit message应该包括三个部分：操作类型、作用域、具体信息。
举个例子，当我给项目新添加了一个功能，我在commit message上应该写上

```bash
feat(XXX): added a new feature to XXX
```

## 操作类型
这里有一个列表，列出了几乎所有操作类型：


| 操作类型 | 说明 |  
| -------- | -------- |
| feat | 新增功能 |  
| fix | 修复bug |
| refactor | 重构，既不增加新的功能，也不修改任何bug |
| docs | 增加或修改文档，比如README.md |
| style | 修改代码风格，但不影响功能 |
| test | 添加测试用例，或者修改测试 |
| chore | 杂项，比如修改或添加.gitignore文件、cmake文件等构件脚本 |
| perf | 性能优化 |
| ci | CI/CD相关的改动 |
| build | 构建方式或者依赖的改动 |
| revert | 回滚某个提交 |


## 作用域
一般说清楚是哪个模块或者哪一层即可，比如修改了一个项目的路由，作用域可以写为route

## 具体信息
尽量详细地描述具体的改动，方便后续的追踪和理解