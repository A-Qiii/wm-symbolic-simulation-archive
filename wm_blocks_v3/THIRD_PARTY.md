# External resources and attribution

- CALVIN environment: `https://github.com/mees/calvin_env.git`, commit
  `1431a46bd36bde5903fb6345e68b5ccc30def666`, MIT. The original parent CALVIN
  checkout was `fa03f01f19c65920e18cf37398a9ce859274af76`; only its pinned
  environment submodule is required here. `licenses/CALVIN_MIT.txt` preserves
  the notice for the adapted primitive block URDFs. Robot/table meshes remain
  external, with upstream notices preserved by the resource checkout.
- The bundled `assets/large.urdf` and `assets/small.urdf` adapt the upstream
  blue/red box dimensions, mass and inertia. `configs/assets.json` records
  source checksums and world sizes. There is no bundled robot mesh or dataset.
- DINOv2: Hugging Face `facebook/dinov2-small`, revision
  `ed25f3a31f01632728cabb09d1542f84ab7b0056`. The pinned model card declares
  Apache-2.0. Weights and model card are downloaded separately, never included
  in this source snapshot. Checksums are in `configs/external_resources.json`.
- Python packages retain their individual upstream licenses. Exact workload
  dependency closures and tested Python versions are recorded under
  `requirements/` and `reproducibility/`.

No blanket license has been selected for the project's own code in this
preparation snapshot; the upstream notices above do not license all project
code automatically.
