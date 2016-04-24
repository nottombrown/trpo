# TRPO with continuous actions

This repo implements a TRPO agent ( http://arxiv.org/abs/1502.05477 ) by modifying https://github.com/wojzaremba/trpo and replacing the softmax distributions with Gaussian distributions, and adding a tiny bit of bells and whistles.

To run the code, simply type python main.py --task $TASK_NAME.  Once training is complete, the code upload the run using your OpenAI gym account. 





