Recurrent Fully Convolutional Networks
======================================

.. image:: https://app.wercker.com/status/08d2ff090d6908d3b2f234d103701ef6/s/master
   :target: https://app.wercker.com/project/byKey/08d2ff090d6908d3b2f234d103701ef6

Recurrent Fully Convolutional Networks for Instance-level Object Segmentation.


Installation
------------

.. code-block:: bash

  mkdir -p ~/rfcn_ws/src
  cd ~/rfcn_ws

  virtualenv venv
  . venv/bin/activate

  cd ~/rfcn_ws/src

  git clone https://github.com/pdollar/coco.git
  (cd coco/PythonAPI && python setup.py install)

  git clone https://github.com/wkentaro/rfcn.git
  (cd rfcn && python setup.py develop)


Testing
-------

Currently we have linter checking.

.. code-block:: bash

  pip install flake8 hacking
  flake8 .


----


Dataset
-------

- COCO: http://mscoco.org
- Pascal VOC2012: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/


Related Works
-------------


Recurrent Instance Segmentation
+++++++++++++++++++++++++++++++

- ECCV 2016: https://arxiv.org/pdf/1511.08250v3.pdf


Translation-aware Fully ConvolutionalInstance Segmentation
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

- 2016: http://image-net.org/challenges/talks/2016/ta-fcn_coco.pdf


Fully Convolutional Networks
++++++++++++++++++++++++++++

- 2015: http://cvn.ecp.fr/personnel/iasonas/slides/FCNNs.pdf
- 2015: https://arxiv.org/pdf/1605.06211v1.pdf
- https://github.com/wkentaro/fcn


Recurrent Fully Convolutional Networks
++++++++++++++++++++++++++++++++++++++

- 2016: https://arxiv.org/pdf/1606.00487v3.pdf


Proposal-free Network for Instance-level Object Segmentation
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

- 2015: https://arxiv.org/pdf/1509.02636v2.pdf


Reversible Recursive Instance-level Object Segmentation
+++++++++++++++++++++++++++++++++++++++++++++++++++++++

- 2016: http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Liang_Reversible_Recursive_Instance-Level_CVPR_2016_paper.pdf
