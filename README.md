# Taichi_FEM_DEM_doubleLayerRepulsion

### 团队名：我在搓泥巴

### 项目名：颗粒间双电层斥力求解与作用

### 项目介绍：

胶体体系（如黏土、牛奶、人体血液等）中颗粒的作用力主要为以下三种：范德华吸引力、双电层排斥力、物理接触力。

（1）求解过程：颗粒之间的双电层斥力，普遍采用有限单元法对Poisson-Boltzmann方程（PB方程）进行求解来获得。

（2）运动过程：颗粒在双电层斥力下的运动，则需要将上述PB方程求解的结果进行拟合或者简化为“力与距离”的表达式，代入到离散单元法中进行求解。

在当前的研究中，（1）求解过程与（2）运动过程是分开实现，且对PB方程的结果进行拟合或简化会存在一定的误差，因此本项目拟使用Taichi，在每个时间步内通过有限单元法对颗粒之间的双电层斥力进行求解，并使用离散单元法对颗粒的速度与位置进行更新。


#### 参考资料

有限单元法求解双电层排斥力：

1991-Numerical Study of The Electrical Double-Layer Repulsion Between Nonparallel Clay Particles of Finite Length[J]. International Journal for Numerical and Analytical Methods in Geomechanics[doi:10.1002/nag.1610151002](https://onlinelibrary.wiley.com/doi/10.1002/nag.1610151002)

2020-Numerical Studies on Electrical Interaction Forces and Free Energy between Colloidal Plates of Finite Size[J]. LANGMUIR [doi:10.1021/acs.langmuir.9b02981](https://pubs.acs.org/doi/10.1021/acs.langmuir.9b02981)

离散单元法应用双电层斥力：

2003-Three-Dimensional Discrete Element Method of Analysis of Clays[J]. Journal of Engineering Mechanics [doi:10.1061/(asce)0733-9399(2003)129:6(585)](https://ascelibrary.org/doi/10.1061/%28ASCE%290733-9399%282003%29129%3A6%28585%29)

2022-Discrete-element simulation of drying effect on the volume and equivalent effective stress of kaolinite[J]. Géotechnique [doi:10.1680/jgeot.18.P.239](https://www.icevirtuallibrary.com/doi/full/10.1680/jgeot.18.P.239)
