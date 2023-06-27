---
title: Markdown语法备忘录
date: '2022-6-27'
tags: ['Markdown']
draft: false
summary: "Markdown 是一种轻量级标记语言，创始人为约翰·格鲁伯（John Gruber）。 它允许人们使用易读易写的纯文本格式编写文档，然后转换成有效的 XHTML（或者HTML）文档。这种语言吸收了很多在电子邮件中已有的纯文本标记的特性。由于 Markdown 的轻量化、易读易写特性，并且对于图片，图表、数学式都有支持，许多网站都广泛使用 Markdown 来撰写帮助文档或是用于论坛上发表消息。 如 GitHub、Reddit、Diaspora、Stack Exchange、OpenStreetMap 、SourceForge、简书等，甚至还能被使用来撰写电子书。"
---

Markdown标题
---
```markdown
一级标题
========
二级标题
--------
# 一级标题
## 二级标题
### 三级标题
#### 四级标题
##### 五级标题
###### 六级标题
```
## Markdown段落
```markdown
这是第一段(两个空格)  
这是第二段

这是第三段
```
## 字体
```markdown
*斜体*
_斜体_
**粗体**
__粗体__
***粗斜体***
___粗斜体___
```
## 分隔线
```markdown
***
* * *
******
- - -
--------
```
## 删除线
```markdown
~~删除线~~
```
## 下划线
```markdown
<u>HTML</u>
```
## 脚注
```markdown
这有一个脚注 [^脚注]

[^脚注]:这是脚注
```
  
## Markdown列表
```markdown
* 第一项
* 第二项
* 第三项
------ 分割线 ------
+ 第一项
+ 第二项
+ 第三项
------ 分割线 ------
- 第一项
- 第二项
- 第三项
```
```markdown
1. 第一项
2. 第二项
3. 第三项
```
### 列表嵌套
```markdown
1. 第一项：
    - 第一项嵌套的第一个元素
    - 第一项嵌套的第二个元素
2. 第二项：
    - 第二项嵌套的第一个元素
    - 第二项嵌套的第二个元素
```
## Markdown区块
Markdown 区块引用是在段落开头使用 **>** 符号 ，然后后面紧跟一个**空格**符号：
```markdown
> 区块引用
> 菜鸟教程
> 学的不仅是技术更是梦想
```
### 区块中使用列表
```markdown
> 区块中使用列表
> 1. 第一项
> 2. 第二项
> + 第一项
> + 第二项
> + 第三项
```
### 列表中使用区块
如果要在列表项目内放进区块，那么就需要在 **>** 前添加四个空格的缩进。
```markdown
* 第一项
    > 菜鸟教程
    > 学的不仅是技术更是梦想
* 第二项
```
## Markdown代码
### 单行代码
```markdown
`printf()` 函数
```

### 代码区块
代码区块使用三个(或以上)反引号对**```** 包裹一段代码，并指定一种语言（也可以不指定）：
````Markdown
```python
class CaseData:
	pass
```
````
## Markdown表格
```markdown
|  表头   | 表头  |
|  ----  | ----  |
| 单元格  | 单元格 |
| 单元格  | 单元格 |
```
#### 对齐方式
-   **-:** 设置内容和标题栏居右对齐。
-   **:-** 设置内容和标题栏居左对齐。
-   **:-:** 设置内容和标题栏居中对齐。
```markdown
| 左对齐 | 右对齐 | 居中对齐 | 
| :-----| ----: | :----: | 
| 单元格 | 单元格 | 单元格 | 
| 单元格 | 单元格 | 单元格 |
```
| 左对齐 | 右对齐 | 居中对齐 |
| :----| ----: | :-----: |
| 单元格 | 单元格 | 单元格 |
| 单元格 | 单元格 | 单元格 |

## Markdown高级技巧
### 支持的HTML元素
```markdown
使用 <kbd>Ctrl</kbd>+<kbd>Alt</kbd>+<kbd>Del</kbd> 重启电脑
```
使用 <kbd>Ctrl</kbd>+<kbd>Alt</kbd>+<kbd>Del</kbd> 重启电脑
### 转义
```markdown
**文本加粗** 
\*\* 正常显示星号 \*\*
```
Markdown 支持以下这些符号前面加上反斜杠来帮助插入普通的符号：
```markdown
\   反斜线
`   反引号
*   星号
_   下划线
{}  花括号
[]  方括号
()  小括号
#   井字号
+   加号
-   减号
.   英文句点
!   感叹号
```
## 公式
当你需要在编辑器中插入数学公式时，可以使用两个美元符`$$`包裹**TeX**或**LaTeX**格式的数学公式来实现。提交后，问答和文章页会根据需要加载 Mathjax 对数学公式进行渲染。如果是在博客园用公式，需要到自己的博客园：管理》选项，勾选“启用数学公式支持”；如果是在typora软件：文件》偏好设置》markdown》扩展语法，勾选“内联公式”；其他的md编辑器应该也是差不多需要去设置渲染公式，不然显示不出来。

公式的用法太多了，需要的可以自行去研究，这个挺全的[https://www.zybuluo.com/codeep/note/163962#mjx-eqn-eqsample](https://www.zybuluo.com/codeep/note/163962#mjx-eqn-eqsample)，下面列举一个比较简单的：
```markdown
$$ J_\alpha(x) = \sum_{m=0}^\infty \frac{(-1)^m}{m! \Gamma (m + \alpha + 1)} {\left({ \frac{x}{2} }\right)}^{2m + \alpha} \text {，独立公式示例} $$
```
$$ J_\alpha(x) = \sum_{m=0}^\infty \frac{(-1)^m}{m! \Gamma (m + \alpha + 1)} {\left({ \frac{x}{2} }\right)}^{2m + \alpha} \text {，独立公式示例} $$

## 文字颜色、大小、字体
#### 颜色
```markdown
<font color="#dd0000">文字颜色预览</font>
```
<font color="#dd0000">文字颜色预览</font>
#### 大小
```markdown
size为1：<font size="1">size为1</font>
size为2：<font size="2">size为2</font>
size为3：<font size="3">size为3</font>
size为4：<font size="4">size为4</font>
size为6：<font size="6">size为6</font>
```
size为1：<font size="1">size为1</font>
size为2：<font size="2">size为2</font>
size为3：<font size="3">size为3</font>
size为4：<font size="4">size为4</font>
size为6：<font size="6">size为6</font>
#### 字体
```markdown
<font face="黑体">我是黑体字</font> 
<font face="宋体">我是宋体字</font> 
<font face="楷体">我是楷体字</font> 
<font face="微软雅黑">我是微软雅黑字</font> 
<font face="fantasy">我是fantasy字</font>
<font face="Helvetica">我是Helvetica字</font>
```
<font face="黑体">我是黑体字</font> 
<font face="宋体">我是宋体字</font> 
<font face="楷体">我是楷体字</font> 
<font face="微软雅黑">我是微软雅黑字</font> 
<font face="fantasy">我是fantasy字</font>
<font face="Helvetica">我是Helvetica字</font>

#### 背景色
```markdown
<table><tr><td bgcolor=#F4A460>背景色的设置是按照十六进制颜色值：#F4A460</td></tr></table>
<table><tr><td bgcolor=#FF6347>背景色的设置是按照十六进制颜色值：#FF6347</td></tr></table>  
<table><tr><td bgcolor=#D8BFD8>背景色的设置是按照十六进制颜色值：#D8BFD8</td></tr></table>  
<table><tr><td bgcolor=#008080>背景色的设置是按照十六进制颜色值：#008080</td></tr></table>  
<table><tr><td bgcolor=#FFD700>背景色的设置是按照十六进制颜色值：#FFD700</td></tr></table>
```

## Emoji
emoji使用时复制后面的md代码即可（包括两个冒号），也可以直接复制emoji表情