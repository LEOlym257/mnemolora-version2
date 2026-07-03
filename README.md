# AdaJEPA: An Adaptive Latent World Model

**Abstract:** Latent world models enable planning from high-dimensional observations by predicting future states in a compact latent space. However, these models are typically kept frozen at test time: when their predictions become inaccurate, planning can fail, especially under test-time distribution shift. To address this, we propose AdaJEPA, an adaptive latent world model that performs test-time adaptation within the closed loop of model predictive control (MPC). After training, AdaJEPA plans and executes the first action chunk, uses the observed next-state transition as a self-supervised adaptation signal, and replans with the updated model. This closed-loop update continuously recalibrates the world model without additional expert demonstrations. Across a range of goal-reaching tasks, AdaJEPA substantially improves planning success with as few as one gradient step per MPC replanning step.

<p align="center">
  &#151; <a href="https://agenticlearning.ai/adajepa/"><b>View Paper Website</b></a> &#151;
  <a href="https://arxiv.org/abs/2606.32026"><b>View Paper</b></a> &#151;
</p>

![main_loop](assets/main_loop.png)

## Code

The authors are currently at ICML and will release the full code after ICML (Sorry about the delay!). Thank you for your patience and interest!

## Citation

```bibtex
@misc{wang2026adajepaadaptivelatentworld,
      title={AdaJEPA: An Adaptive Latent World Model}, 
      author={Ying Wang and Oumayma Bounou and Yann LeCun and Mengye Ren},
      year={2026},
      eprint={2606.32026},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2606.32026}, 
}
```
