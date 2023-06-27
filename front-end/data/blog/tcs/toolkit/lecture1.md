---
title: TCS Toolkit学习笔记(1)
date: '2022-6-27'
tags: ['TCS']
draft: false
summary: "介绍了TCS课程的主要内容，分享了如何学好TCS的经验，纠正了一些错误的Latex语法"
---

## About the course
### TCS Toolkit
1. TCS = Theoretical Computer science = Algorithms and Computational Complexity = STOC/FOCS topics
2. Toolkit: Intro to tools and topics that arise in TCS research
    > About 50% math topics, 50% CS topics
## How to TCS 
### Stay au courant
1. Read TCS blogs, announcements: http://cstheory-feed.org
2. Follow your favorite researchers on Twitter
3. Watch videos of talks from elsewhere:
    + https://sites.google.com/site/plustcs/
    + https://video.ias.edu/csdm
    + https://simons.berkeley.edu/videos
    + Many conferences (e.g., some STOC/FOCS) have videos of the talks available online
    + Youtube channels:
        CMU Theory, TCS+, Princeton TCS, Shannon Channel, Simon Institute, ...
### Find TCS papers to read
1. Proceedings of recent FOCS/STOC/SODA/CCC
2. Recent posts to arXiv (https://arxiv.org/archive/cs) or ECCC (https://eccc.weizmann.ac.il)
3. Papers citing / cited by the paper you're reading


### Managing references
1. Saave a local copy of all papers you ever look at
2. Use a consistent naming convention
    E.g., nisan-wigderson-log-rank-conj.pdf
3. Maintain a lifetime .bib file for all papers you ever reference
4. Use a consistent BibTex key style. E.g. \{NW93\}
5. Use reference managment software: JabRef, BibDesk, ...


### Writing math: LaTex
1. You need a great text editor
2. You need version-control
3. Use indentation!
4. Create a lifetime stub .tex file and a lifetime .sty file


### LaTex - lifetime .bib file
+ Where to get .bib entries:
    1. Always try https://mathscinet.ams.org/mrlookup first
### LaTEX - my top peeves

```latex
% Don't                    % Do
$ < U, V> $                $ \langle U, V \rangle $

"quotes"                   ``quotes"

$ log(1+x) $               $ \log(1+x) $

\[                         \[
	(\frac{ax+b}{cy})^2        \left(\frac{ax+b}{cy}\right)^2  
\]                         \]

If A is a matrix, then     If $A$ is a matrix, then

we execute $ALG(x)$        we execute $\textnormal{ALG}(x)$

\begin{eqnarray}           \begin{align}
	y &=& (x+1)^2 \\           y $= (x+1)^2 \\
	  $=$ x^2+x2+1               $= x^2+2x+1
\end{eqnarray}             \end{align}

Lemma \ref{lem:big} is     Lemma~\ref{lem:big} is 
due to Blum \cite{Blu99}   due to Blum~\cite{Blu99}

one party, e.g. Alice, is one party, e.g.\ Alice, is

\begin{proof}              \begin{proof}
\[                         \[
	x=1 \implies x^2=1.        x=1 \implies x^2=1. \qedhere
\]                         \]
\end{proof}                \end{proof}
```

### LaTex - drawing
1. To draw figures: *Inkscape*
## Street Fighting Mathematics
1. Wikipedia https://www.wikipedia.org/
2. https://oeis.org
3. Inverse Symbolic Calculator https://wayback.cecm.sfu.ca/projects/ISC/ISCmain.html
4. mathoverflow  https://mathoverflow.net/
5. Stackexchange sites
    + https://mathoverflow.net/
    + https://cstheory.stackexchange.com/
    + https://cs.stackexchange.com/
    + https://math.stackexchange.com/
    + https://tex.stackexchange.com/
6. Ask Mathematica/Maple/Sage.
