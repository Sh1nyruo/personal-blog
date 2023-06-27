---
title: So-VITS原理及应用
date: '2022-6-27'
tags: ["Python", "Pytorch", "Colab", "AI", "VAE"]
draft: false
summary: "语音合成（TTS）是当下最火热的深度学习应用之一，这项技术可以用来制作语音助手、发布语音信息、以及为视障人士提供文本内容的语音播报等等。它可以为人们提供一种更加直观、便捷的信息获取方式，帮助人们更有效地理解和使用信息。

在某些情况下，人们可能不方便或不能直接阅读文本内容，比如在开车的时候、在阅读困难的情况下。这时候，AI语音合成技术就可以派上用场，帮助人们轻松获取信息。

除了在商业领域应用以外，TTS技术如今也流行于媒体领域，大到企业，小到个人，都可以使用TTS技术制作独特的虚拟声音。"
---


##### 顾永威 2020111475
## VITS语音合成模型实践
+ 参考论文
    > Kim, Jaehyeon, Jungil Kong, and Juhee Son. "Conditional variational autoencoder with adversarial learning for end-to-end text-to-speech." _International Conference on Machine Learning_. PMLR, 2021.
    > [论文地址](https://proceedings.mlr.press/v139/kim21f.html)
+ 项目实践
    > AI七海Nana7mi
    > [七海Nana7mi的BiliBili个人空间](https://space.bilibili.com/434334701?spm_id_from=333.337.0.0)
    > [BiliBili投稿展示](https://www.bilibili.com/video/BV1TG4y1J75F/)
    
## 项目意义
语音合成（TTS）是当下最火热的深度学习应用之一，这项技术可以用来制作语音助手、发布语音信息、以及为视障人士提供文本内容的语音播报等等。它可以为人们提供一种更加直观、便捷的信息获取方式，帮助人们更有效地理解和使用信息。

在某些情况下，人们可能不方便或不能直接阅读文本内容，比如在开车的时候、在阅读困难的情况下。这时候，AI语音合成技术就可以派上用场，帮助人们轻松获取信息。

除了在商业领域应用以外，TTS技术如今也流行于媒体领域，大到企业，小到个人，都可以使用TTS技术制作独特的虚拟声音。
## 已有的研究工作
TTS（TEXT TO SPEECH）技术发展已久，古老到火车站的播报，近到前两年火爆的有声小说，TTS技术已经在实际生活中随处可见。

现代TTS一般分为两种模型：
1. one-stage system：一般称为**端到端（end-to-end）**，即直接从**文本**到**波形**
2. two-stage system：即先从**文本**到**梅尔谱/线性谱**，再将**梅尔谱/线性谱**生成**波形**

过去几年，two-stage system比较流行，每个阶段都有许多模型被提出。基于神经网络的自回归的TTS系统（Neural network-based autoregressive TTS systems）展现了其在合成逼真音频的能力，但该模型的推理速度太慢，为了克服该缺点，有很多非自回归的模型被提出 ：
+ 在**文本-频谱（text-to-spectrogram）**阶段，提取预训练自回归模型的注意力图（attention maps）降低文本与频谱的对齐难度。除此之外，最近也有一些基于似然的方法，可以避免使用外部的对齐工具，最大化梅尔谱的似然度。
+ 在**频谱-波形（spectrogram-to-waveform）**阶段，生成式对抗网络（Generative Adversarial Networks）展现了良好的性能。

尽管一直有新的模型被提出，但两阶段模型存在致命的问题——无法生成高质量的音频。由于一二阶段都是分开训练的，一阶段**文本-频谱**训练得到的频谱并不是真实频谱，而二阶段**频谱-波形**参与训练的都是真实的频谱，因此，两阶段模型始终存在缺陷。

由于两阶段模型的缺陷，近年来一直有end-to-end的TTS模型被提出，比如FastSpeech 2s、EATS等，但是效果都不如两阶段模型。

直到2021年本论文提出，本论文使用了基于cVAE+Flow+GAN的新的VITS模型，将TTS推向了新的阶段。
## 一些基础知识
+ <font  size="4">KL-DIVERGENCE</font>
    + KL散度可以用来**衡量两个概率分布之间的相似度**。
    假设两概率分布$p(x),q(x)$，则$p$对$q$的KL散度为：
    $$
  \begin{aligned}
	D_{K L}(p \| q) & =H(p, q)-H(p) \\
    & =-\sum_x p(x) \log q(x)-\sum_x-p(x) \log p(x) \\
    & =-\sum_x p(x)(\log q(x)-\log p(x)) \\
    & =-\sum_x p(x) \log \frac{q(x)}{p(x)}
  \end{aligned}
  $$
  其中，$H(p, q)=\sum_x p(x) \log \frac{1}{q(x)}=-\sum_x p(x) \log q(x)$
  **KL散度**的性质：
    1. $D_{K L}(P \| Q) \neq D_{K L}(Q \| P)$
    2. $D_{K L}(P \| Q) \geqslant 0 \text { ， 仅在 } P=Q \text { 时等于 } 0$
+ <font size="4">VAE (Variational Autoencoder)</font>
    + **Auto-encoder**是一种**无监督**的NN模型，分为**Encoder**和**Decoder**两个NN模型，Encoder部分负责学习输入数据的隐含特征，Decoder负责将隐含特征重构成原始输入数据。Auto-encoder的目的是使Decoder得到的数据尽可能地接近Encoder的输入数据
    在训练完一个**Auto-encoder**后，我们可以得到一个**Encoder**和一个**Decoder**，不妨仔细观察**Decoder**，输入一段编码，输出一段数据，那么是否可以直接输入一串随机的编码让**Decoder**输出对应的数据呢？例如一张图片、一段音频，可惜这样的效果并不好，因此我们需要一些新的方法。
    **Variational Autoencoder**是在2013年首次提出的一种**Generative Model**。在一般的**Auto-encoder**中，**Encoder**会直接产生一个编码，记作$(m_1,m_2,\ldots ,m_k)$，但是在VAE中，为了给该编码添加合适的噪音，**Encoder**除了输出$(m_1,m_2,\ldots ,m_k)$，还会输出一个**噪音权重编码**$(\sigma_1,\sigma_2,\ldots ,\sigma_k)$，同时产生一个**噪音**$(e_1,e_2,\ldots ,e_k)$，最终输入到**Decoder**的编码就是$(c_1,c_2,\ldots ,c_k)$，其中$c_i=\exp \left(\sigma_i\right) \times e_i+m_i$。
    $Loss Function$除了必要的**重构损失**，还需要一个损失函数控制噪音大小，该额外的$Loss Function=\sum_{i=1}^k\left(\exp \left(\sigma_i\right)-\left(1+\sigma_i\right)+\left(m_i\right)^2\right)$，推导见下。
    VAE作为生成模型（Generative Model），其理论基础为高斯混合模型。若我们将编码定义为一个连续随机遍历$z$，假设$z$服从正态分布$N(0,1)$，对于每个采样（编码）$z$，则原始分布$P(x)=\int_z P(z) P(x \mid z) d z$，其中$z \sim N(0,1), \quad x \mid z \sim N(\mu(z), \sigma(z))$。
    接下来利用极大似然估计，由于上式难以求解，于是可等价求解以下方程：$$\text { Maximum } L=\sum_x \log P(x)$$
    注意到
    $$
    \begin{aligned}
    \log P(x)& =\int_z q(z \mid x) \log P(x) d z \quad(q(z \mid x) 可以是任何分布 )\\
    & =\int_z q(z \mid x) \log \left(\frac{P(z, x)}{P(z \mid x)}\right) d z\\
    & =\int_z q(z \mid x) \log \left(\frac{P(z, x)}{q(z \mid x)} \frac{q(z \mid x)}{P(z \mid x)}\right) d z\\
    & =\int_z q(z \mid x) \log \left(\frac{P(z, x)}{q(z \mid x)}\right) d z+\int_z q(z \mid x) \log \left(\frac{q(z \mid x)}{P(z \mid x)}\right) d z\\
    & =\int_z q(z \mid x) \log \left(\frac{P(z, x)}{q(z \mid x)}\right) d z+K L(q(z \mid x) \| P(z \mid x))
    \end{aligned}
    $$
     由于$K L(q(z \mid x) \| P(z \mid x))\geqslant 0$，
     所以$\log P(x) \geq \int_z q(z \mid x) \log \left(\frac{P(x \mid z) P(z)}{q(z \mid x)}\right) d z$，
     于是原式$\log P(x)=\int_z q(z \mid x) \log \left(\frac{P(x \mid z) P(z)}{q(z \mid x)}\right) d z+K L(q(z \mid x) \| P(z \mid x))$，
     令$L_b=\int_z q(z \mid x) \log \left(\frac{P(x \mid z) P(z)}{q(z \mid x)}\right) d z$
     不难看出$L_b$就是$\log P(x)$的下界，因此求解$\text { Maximum } \log P(x)$等价于求解$\text { Maximum } L_b$
     $$
     \begin{gathered}
	L_b=\int_z q(z \mid x) \log \left(\frac{P(z, x)}{q(z \mid x)}\right) d z \\
	=\int_z q(z \mid x) \log \left(\frac{P(x \mid z) P(z)}{q(z \mid x)}\right) d z \\
	=\int_z q(z \mid x) \log \left(\frac{P(z)}{q(z \mid x)}\right) d z+\int_z q(z \mid x) \log P(x \mid z) d z \\
	=-K L(q(z \mid x)|| P(z))+\int_z q(z \mid x) \log P(x \mid z) d z
	\end{gathered}
	$$
	其中$$-K L(q(z \mid x)|| P(z))=\sum_{i=1}^k\left(\exp \left(\sigma_i\right)-\left(1+\sigma_i\right)+\left(m_i\right)^2\right)$$
	$$
	\begin{aligned}
	& \text { Maximum } \int_z q(z \mid x) \log P(x \mid z) d z \\
	& =\text { Maximum } E_{q(z \mid x)}[\log P(x \mid z)]
	\end{aligned}
	$$
	$-K L(q(z \mid x)|| P(z))+\int_z q(z \mid x) \log P(x \mid z) d z$一般被称为Evidence Lower Bound，以下简称为ELBO，$VAE$的$Loss Function$实际上就是$-ELBO$。
+ <font  size="4">GAN (Generative Adversarial Networks)</font>
    + GAN由两部分构成：生成器（Generator）和判别器（Discriminator），生成器负责生成数据，判别器负责区分生成器生成的数据和真实数据。生成器的目标是欺骗判别器导致其无法区分真假数据，判别器的目标是区分真假数据，GAN就通过这种“博弈”过程，最终生成器和判别器达到一种纳什均衡。

## 项目设计
+ 题目：Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech
+ 原作者：Jaehyeon Kim, Jungil Kong, Juhee Son
+ 项目方法
    > Variational Inference with adversarial learning for end-to-end Text-to-Speech (VITS)
    1. Variational Inference
        + VITS是一种条件变分自编码器（Conditional VAE），最大化目标从$\log P(x)$变为$\log p_\theta(x \mid c)$，其中$c$可能是一段文本，基于类似的推理，不难得到
            $$\log p_\theta(x \mid c) \geq \mathbb{E}_{q_\phi(z \mid x)}\left[\log p_\theta(x \mid z)-\log \frac{q_\phi(z \mid x)}{p_\theta(z \mid c)}\right]$$ 
            其中$p_\theta(x \mid c)$表示隐变量（latent variable）$z$在条件$c$下的先验分布，$p_\theta(x \mid z)$表示数据点$x$的似然方程，即解码器（Decoder），$q_\phi(z \mid x)$是**近似**的后验分布。该cVAE的training loss就是$-ELOB$，也就是**重构损失**$-\log p_\theta(x \mid z)$与**KL-Divergence** $\log \frac{q_\phi(z \mid x)}{p_\theta(z \mid c)}$之和。
        + 重构损失（Reconstruction Loss）
            因为我们的数据$x$是一段音频，为了提高人耳对生成音频的感知质量，因此我们使用梅尔谱代替原始波形，记为$x_{mel}$，我们对隐变量$z$进行上采样，得到一个波形$\hat{y}$，再将$\hat{y}$转换为梅尔谱，记为$\hat{x}_{mel}$，那么我们的重构损失可以记为：$$L_{\text {recon }}=\left\|x_{\text {mel }}-\hat{x}_{\text {mel }}\right\|_1$$
            注意，梅尔频谱仅仅用于重构损失的计算，而不参与变分推断过程。 
        + KL-Divergence
            先验编码器的输入条件$c$包括了一段文本信息，记作$c_{text}$，还有一个对齐矩阵（alignment）$A$，$A$表示了文本信息和频谱之间的对应关系。$A$是一个单调硬对齐矩阵（hard monotonic attention matrix），形状为$\left|c_{\text {text }}\right| \times|z|$。为了给后验编码器提供更多信息，我们使用线性谱作为后验编码器的输入，因此，KL Divergence可以表示为
            $$\begin{array}{r}L_{k l}=\log q_\phi\left(z \mid x_{l i n}\right)-\log p_\theta\left(z \mid c_{\text {text }}, A\right) \\z \sim q_\phi\left(z \mid x_{l i n}\right)=N\left(z ; \mu_\phi\left(x_{l i n}\right), \sigma_\phi\left(x_{l i n}\right)\right)\end{array}$$
            因为我们通常假设$p_\theta\left(z \mid c_{\text {text }}, A\right)$符合高斯分布，这种假设会导致先验分布的表达能力变弱，为了提高先验分布的表达能力，可以引入标准流（Normalizing flow）。标准流可以通过一系列可逆的变换将一个简单的分布变成一个很复杂的分布，由于标准流是可逆的，因此也可以同样将一个复杂的分布变换成一个简单的分布。我们使用标准流将假设的先验分布转变为一个更复杂的分布，以此提高先验分布的表达能力。标准流变换如下：
            $$\begin{aligned}p_\theta(z \mid c) & =N\left(f_\theta(z) ; \mu_\theta(c), \sigma_\theta(c\right\left|\operatorname{det} \frac{\partial f_\theta(z)}{\partial z}\right| \\c & =\left[c_{t e x t}, A\right]\end{aligned}$$
            经过标准流变换后，原本假设的先验分布由高斯分布变换为了一个更复杂的分布，能生成更逼真的音频。
    2. Alignment Estimation
        + 单调对齐搜索算法（MONOTONIC ALIGNMENT SEARCH，MAS）
            为了估计文本和语音之间的对齐矩阵$A$，VITS利用了单调对齐搜索算法。为了搜索到一个最优的对齐路径，可以利用极大似然估计，得到$$A=\underset{\hat{A}}{\arg \max } \log p\left(x \mid c_{\text {text }}, \hat{A}\right)$$
            这个算法要求对齐矩阵$A$是单调且非跳跃的。但是MAS算法是应用于Glow模型，而不适用于VITS模型，因为Glow模型的目标是确切的对数似然函数，而VITS模型的优化目标是$ELBO$。因此，我们需要将MAS的最大化目标改为$ELOB$，即
            $$\begin{aligned}& \underset{\hat{A}}{\arg \max } \log p_\theta\left(x_{m e l} \mid z\right)-\log\frac{q_\phi\left(z \mid x_{l i n}\right)}{p_\theta\left(z \mid c_{\text {text }}, \hat{A}\right)} \\& =\underset{\hat{A}}{\arg \max } \log p_\theta\left(z \mid c_{\text {text }}, \hat{A}\right) \\& =\logN\left(f_\theta(z) ; \mu_\theta\left(c_{t e x t}, \hat{A}\right), \sigma_\theta\left(c_{t e xt}\hat{A}\right)\right) \\&\end{aligned}$$   
        
        + DURATION PREDICTION FROM TEXT
            我们能为每个输入$d_i$计算持续时间，即$\sum_j A_{i, j}$，其中$i$表示第$i$个输入文本，$j$表示第$j$个对齐矩阵，利用这些数据可以训练一个固定时长预测器，但是这种方式得到的输出音频缺乏多样性，为了提高输出的多样性，我们采用随机时长预测器（stochastic duration predictor），随机时长预测器是一中基于流（Flow）的生成模型，既然是生成模型，就要用极大似然估计训练，但直接训练存在困难：
            1. 时长是离散的，但Flow模型要求输入是连续的，因此需要去量化（dequantize）
            2. 时长是一个标量，不能实现高维可逆变化。
            
            因此，VITS使用了**variational dequantization**和**variational data augmentation**两种技术。
            设时长序列为$d$，则$d$中的每个元素都应大于1（因为MAS不允许跳过）。引入序列$u$，它和$d$有相同维度，且每个元素取值范围为$[0,1)$，这样$d-u$的每个元素都是浮点数且为正实数。另一边，引入序列$v$，它与$d$有相同的时间精度，将它与$d-v$拼接起来就可以获得高维的序列。我们从$q_\phi\left(u, \nu \mid d, c_{t e x t}\right)$中采样$u,v$，最终我们随机时长的变分下界似然就是
            $$
            \begin{aligned}
		    & \log p_\theta\left(d \mid c_{\text {text }}\right) \geq \\
		    & \mathbb{E}_{q_\phi\left(u, \nu \mid d, c_{\text {text }}\right)}\left[\log \frac{p_\theta\left(d-u, \nu \mid c_{\text {text }}\right)}{q_\phi\left(u, \nu \mid d, c_{\text {text }}\right)}\right]
		    \end{aligned}
            $$
            那么，随机时长预测器的损失$L_{dur}$就是上述下界相反数，注意，随机时长预测器的训练与其他部分隔离，采样方式为给该模型一个噪音输入，通过Flow的逆变换会得到一个表示时长的整数。
    3. Adversarial Training
        + 用于对抗训练的最小二乘损失函数
            1. 生成器G
                $L_{a d v}(G)=\mathbb{E}_z\left[(D(G(z))-1)^2\right]$
            3. 判别器D
                $L_{a d v}(D)=\mathbb{E}_{(y, z)}\left[(D(y)-1)^2+(D(G(z)))^2\right]$
        + 额外用于生成器训练的特征匹配损失（feature-matching loss）
            $L_{f m}(G)=\mathbb{E}_{(y, z)}\left[\sum_{l=1}^T \frac{1}{N_l}\left\|D^l(y)-D^l(G(z))\right\|_1\right]$
            其中$T$代表判别器的总层数，$D^l$表示第$l$层的特征图（每层$N_l$个特征)
    4. 最终的损失函数$Loss Function$
        VITS可以看作是VAE和GAN的联合训练，因此总体损失为：
        $L_{\text {vae }}=L_{\text {recon }}+L_{k l}+L_{d u r}+L_{a d v}(G)+L_{f m}(G)$
    5. 模型结构
        1. 后验编码器
            **仅用于训练阶段，不用于推理阶段**
            在训练时输入线性谱，输出隐变量$z$，推断时$z$由$f_\theta(z)$。VITS的后验编码器采用WaveGlow和Glow-TTS中的非因果WaveNet残差模块。应用于多人模型时，将说话人加入到残差模块中。
        3. 先验编码器
            包含一个文本编码器，将输入文本编码为$c_{text}$，还包括一个标准流$f_\theta$，提高先验分布的表达能力。多人模型同后验编码器。
        5. 解码器
            实际就是声码器HiFi-GAN V1的生成器。多人模型时把说话人信息加入到声码器中。
        7. 判别器
            **仅用于训练阶段，不用于推理阶段**
            实际就是HiFi-GAN的多周期判别器。
        9. 随机时长预测器
            从条件输入$h_{text}$估算音频时长。多人模型同理。
            
+ 模型架构
    + 训练模型
        ![[Pasted image 20221216203520.png]]
    + 合成模型
        ![[Pasted image 20221216203540.png]]
+ 创新点
    1. 成功把Flow-VAE应用到了TTS任务，利用Normalizing Flow提高了cVAE先验编码器的编码能力，效果显著。
    2. 训练简单，不需要two-stage架构，跳过了频谱这一中间步骤，减少了GAP带来的Loss
    3. 使用了随机时长预测器而不是固定时长，使得情感表达多样化，效果非常好。
+ 困难及解决方案
    1. 结合VAE模型和Flow模型
        + 解决方案：将标准流引入VAE模型的先验分布，提高了先验分布的表达能力。
    2. 模型复杂，难以理解
        + 解决方案：学习并证明
    4. 数据集不好搞
        + 解决方案：使用了自动切片器，得到了简陋的数据集
## 实验一 语音合成
> [训练Colab](https://colab.research.google.com/drive/1sdUEScOJZOpmIunV5hy2PGsQwYUxjMr2#scrollTo=p6TJx-R57hZx)
> [生成Colab](https://colab.research.google.com/drive/1sdUEScOJZOpmIunV5hy2PGsQwYUxjMr2#scrollTo=H5rT3we41p0P)
1. 数据集
    + LJ Speech dataset（官方）
3. 训练
    + 先采用原作者使用的单人数据集LJ Speech dataset进行训练，训练平台为[Colab](https://colab.research.google.com/drive/1sdUEScOJZOpmIunV5hy2PGsQwYUxjMr2#scrollTo=p6TJx-R57hZx)
    + GPU：TESLA T4 16GB
    + 训练时长：12小时
5. 结果分析
    + 作为生成器输入的文本
        > *Four score and seven years ago our fathers brought forth on this continent a new nation, conceived in liberty, and dedicated to the proposition that all men are created equal.*
    + 原音频梅尔谱
        ![[Pasted image 20221215214922.png]]
    + $Step = 0$
        ![[Pasted image 20221215215319.png]]
    + $Step = 2000$
        ![[Pasted image 20221215220339.png]]
        音频见<kbd>实验数据/step2000.wav</kbd>
    + $Step = 6000$
        ![[Pasted image 20221215221738.png]]
        音频见<kbd>实验数据/step6000.wav</kbd>
    +  $Step = 11000$
        ![[Pasted image 20221216094221.png]]
        音频见<kbd>实验数据/step11000.wav</kbd>
    +  $Step = 18000$
    + ![[Pasted image 20221216094353.png]]
        音频见<kbd>实验数据/step18000.wav</kbd>
    +  $Step = 30000$
        ![[Pasted image 20221216141625.png]]
        音频见<kbd>实验数据/step30000.wav</kbd>
	+ 由于时间限制，仅跑了76个Epoch，但是可以看出模型已经接近收敛，得到的音频已经能清晰听出内容，仅存在一部分电流音。
## 实验二 音色转换
> [Colab](https://colab.research.google.com/drive/1MlskEj-vf788V609iHNWRB3TG9GnSOTY#scrollTo=Ma2YWTU4afOY)
1. 数据集
    + 七海Nana7mi音频数据集（自己制作i）
2. 训练
    + 使用VITS+Soft-diffusion训练，音色转换训练
    + GPU：A100-SXM4-40GB
    + 训练时长：12小时
5. 结果分析
    + 待转换歌曲
        > 《Killer Queen》 Queen
    + 原音频梅尔谱
        ![[Pasted image 20221216145801.png]]
    + $Step=0$
        ![[Pasted image 20221216145832.png]]
    + $Step=10000$
        ![[Pasted image 20221216175114.png]]
    + Loss
        ![[Pasted image 20221217111512.png]]
        详细数据在<kbd>eval</kbd>中，使用<kbd>tensorborad</kbd>查看
    + 数据集来自[七海Nana7mi](https://space.bilibili.com/434334701)的直播录像数据
        1. 由于数据集不能包含BGM，因此只能选取一般的聊天录像，导致没有高音域，无法唱超高音。
        2. 由于直播录像中难免存在一些环境音，包括但不限于杯子摔倒声、拍手声、车声，因此训练出的效果没有官方数据集那么好，不过比起传统的Vocaloid的合成技术，有明显优势。
        3. 训练多了过拟合了，由于原数据集音频质量并不高，过拟合影响非常大。
        4. 最大的制约不在本模型，而是待转换歌曲的质量，需要纯人声，如果不纯净，那么转换的效果一般，如果纯净，效果则非常好。
        5. 模型效果非常强，如果数据集质量高，并且带转换歌曲的干声干净，那么转换出来的效果就几乎是本人一样，类似效果见<kbd>实验数据/大手拉小手_七海.wav</kbd>。
    + 《Killer Queen》是一首难度极大的歌曲，对于一般歌手来说都难以掌握，但是，这首歌有个好处，人声非常清楚，易于提取干声，因此转换出来的效果比较好。由于成品有点大，所以上传到b站了，见[【AI七海】Killer Queen （Cover Queen）](https://www.bilibili.com/video/BV1TG4y1J75F/?vd_source=02821d2286fef6b5a3a03a1a7c54caa8)
## 结论
+ VITS模型能力十分强大，不仅在单人场景或是多人场景，对其他语音合成模型有显著优势。
+ 但VITS也存在一些问题，它的最终效果比较依赖于数据集的质量，由于不仅需要清晰的音频文件，还需要与音频相对应的文本文件，所以数据集的制作比较麻烦。
+ 使用VITS+Diffusion模型，能强力的实现音色转换，效果非常好，但同时，比较依赖于数据集与带转换音频的质量。
+ 但是这个模型在处理文本时能力较弱，需要进一步改进。
## 参考文献
1. Kim, Jaehyeon, Jungil Kong, and Juhee Son. "Conditional variational autoencoder with adversarial learning for end-to-end text-to-speech." _International Conference on Machine Learning_. PMLR, 2021.
2. B. van Niekerk, M. -A. Carbonneau, J. Zaïdi, M. Baas, H. Seuté and H. Kamper, "A Comparison of Discrete and Soft Speech Units for Improved Voice Conversion," ICASSP 2022 - 2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2022, pp. 6562-6566, doi: 10.1109/ICASSP43922.2022.9746484.
## 参考代码
1. https://github.com/jaywalnut310/vits
2. https://github.com/innnky/so-vits-svc