# vivy——个人学习用cuda库
库的名字出自高三看的动画：vivy-萤石眼之歌(Vivy-Fluorite Eye's Song)

虽然剧情人设上并不特别出彩，但临近本科毕业，又经历种种变故，非常怀念过去的生活，因此想到了这个名字。

设备上使用GTX 1650 laptop运行算子

TODO：

- 计划支持以下算子：HGEMM因为设备问题暂时搁置
  - [ ] SGEMM
  - [ ] reduce
  - [ ] transpose
  - [ ] flash-attention
  - [ ] topk

- 支持使用cuda-event\cupti进行profile
- 支持与cublas比较
- 保存ncu的profile文件
- 记录有优化时候的笔记

# usage
`xmake run transpose` 可以直接运行算子

# 灵感来源
https://github.com/Aoi979/Before-I-Rise 本来只打算与cublas作对比，但看过这位同学的算子库之后惊叹于内容的全面，因此效仿