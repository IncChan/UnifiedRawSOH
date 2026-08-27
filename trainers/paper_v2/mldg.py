"""First-order MLDG for source-only Paper-v2 pseudo-LODO episodes."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Mapping

import torch


MODEL_INPUT_KEYS = (
    "cc_signal",
    "cv_signal",
    "cc_mask",
    "cv_mask",
    "cc_time",
    "cv_time",
    "cc_temperature",
    "cv_temperature",
    "t0_temperature_norm",
)


def model_inputs_from_batch(batch: Mapping[str, Any]) -> dict[str, Any]:
    """Select only current-cycle tensors allowed by the V2 model contract."""

    missing = [key for key in ("cc_signal", "cv_signal") if key not in batch]
    if missing:
        raise ValueError(f"Model batch is missing required current-cycle inputs: {missing}")
    return {key: batch[key] for key in MODEL_INPUT_KEYS if key in batch}


def _call_model(model: torch.nn.Module, batch: Mapping[str, Any]) -> dict[str, Any]:
    inputs = model_inputs_from_batch(batch)
    if hasattr(model, "forward_with_aux"):
        result = model.forward_with_aux(**inputs)
        if not isinstance(result, Mapping) or "soh_pred" not in result:
            raise ValueError("Paper-v2 forward_with_aux must return a mapping containing soh_pred.")
        return dict(result)
    prediction = model(**inputs)
    return {"soh_pred": prediction, "balance_loss": None}


def _zero_loss(reference: torch.Tensor) -> torch.Tensor:
    return torch.zeros((), device=reference.device, dtype=reference.dtype)


def _balance_loss(aux: Mapping[str, Any], reference: torch.Tensor, weight: float) -> torch.Tensor:
    if float(weight) == 0.0:
        return _zero_loss(reference)
    value = aux.get("balance_loss")
    if value is None:
        return _zero_loss(reference)
    if not torch.is_tensor(value):
        raise TypeError("forward_with_aux balance_loss must be a tensor or None.")
    return value


def _functional_forward(
    model: torch.nn.Module,
    parameters: Mapping[str, torch.Tensor],
    batch: Mapping[str, Any],
) -> dict[str, Any]:
    inputs = model_inputs_from_batch(batch)
    buffers = OrderedDict(model.named_buffers())
    try:
        from torch.func import functional_call

        result = functional_call(model, (parameters, buffers), (), inputs)
    except (ImportError, AttributeError):  # pragma: no cover - for older Torch environments
        from torch.nn.utils.stateless import functional_call

        state = OrderedDict()
        state.update(parameters)
        state.update(buffers)
        try:
            result = functional_call(model, state, (), inputs, strict=False)
        except TypeError:  # pragma: no cover - old stateless functional_call signature
            result = functional_call(model, state, (), inputs)
    if isinstance(result, Mapping):
        if "soh_pred" not in result:
            raise ValueError("Functional Paper-v2 forward_with_aux result lacks soh_pred.")
        return dict(result)
    return {"soh_pred": result, "balance_loss": None}


def _gradients(
    loss: torch.Tensor,
    parameters: list[tuple[str, torch.nn.Parameter]],
    *,
    retain_graph: bool = False,
) -> dict[str, torch.Tensor | None]:
    active = [parameter for _, parameter in parameters if parameter.requires_grad]
    if not active or not loss.requires_grad:
        return {name: None for name, _ in parameters}
    values = torch.autograd.grad(
        loss,
        active,
        retain_graph=retain_graph,
        create_graph=False,
        allow_unused=True,
    )
    result: dict[str, torch.Tensor | None] = {}
    position = 0
    for name, parameter in parameters:
        if not parameter.requires_grad:
            result[name] = None
        else:
            result[name] = values[position]
            position += 1
    return result


def _gradient_norm(values: Mapping[str, torch.Tensor | None]) -> float:
    squared = [torch.sum(value.detach() ** 2) for value in values.values() if value is not None]
    if not squared:
        return 0.0
    return float(torch.sqrt(torch.stack(squared).sum()).item())


def first_order_mldg_step(
    model: torch.nn.Module,
    meta_train_batch: Mapping[str, Any],
    pseudo_target_batch: Mapping[str, Any],
    criterion: torch.nn.Module | None = None,
    *,
    inner_learning_rate: float = 1e-3,
    beta: float = 1.0,
    lambda_balance: float = 0.0,
    inner_steps: int = 1,
) -> dict[str, Any]:
    """Compute and assign one first-order MLDG outer gradient.

    The inner update is manual and therefore does not touch an optimizer or
    its momentum/state.  Fast weights are detached after the one inner step;
    pseudo-target gradients are then mapped back to original parameters by
    name.  This is the first-order approximation, not full second-order MAML.
    The caller owns ``optimizer.step()``.
    """

    if int(inner_steps) != 1:
        raise ValueError("Paper-v2 first_order_mldg_step supports exactly one inner step.")
    alpha = float(inner_learning_rate)
    beta = float(beta)
    lambda_balance = float(lambda_balance)
    if alpha <= 0.0:
        raise ValueError("inner_learning_rate must be positive.")
    if beta < 0.0 or lambda_balance < 0.0:
        raise ValueError("beta and lambda_balance must be non-negative.")
    criterion = criterion or torch.nn.MSELoss()
    parameters = [(name, parameter) for name, parameter in model.named_parameters()]
    if not parameters:
        raise ValueError("MLDG model has no parameters.")

    model.zero_grad(set_to_none=True)
    meta_aux = _call_model(model, meta_train_batch)
    meta_prediction = meta_aux["soh_pred"]
    if "soh" not in meta_train_batch:
        raise ValueError("Meta-train batch is missing soh targets.")
    erm_loss = criterion(meta_prediction, meta_train_batch["soh"])
    balance_loss = _balance_loss(meta_aux, erm_loss, lambda_balance)
    inner_loss = erm_loss + lambda_balance * balance_loss
    # The same current-weight ERM graph is also used for the outer ERM term;
    # retain it until that gradient has been collected below.
    inner_grads = _gradients(inner_loss, parameters, retain_graph=True)

    fast_parameters: OrderedDict[str, torch.Tensor] = OrderedDict()
    changed = False
    for name, parameter in parameters:
        if not parameter.requires_grad:
            fast_parameters[name] = parameter
            continue
        gradient = inner_grads[name]
        if gradient is None:
            fast = parameter.detach().clone().requires_grad_(True)
        else:
            fast = (parameter.detach() - alpha * gradient.detach()).requires_grad_(True)
            changed = changed or bool(torch.any(torch.ne(fast.detach(), parameter.detach())).item())
        fast_parameters[name] = fast

    target_prediction_loss: torch.Tensor
    target_grads: dict[str, torch.Tensor | None]
    if beta > 0.0:
        target_aux = _functional_forward(model, fast_parameters, pseudo_target_batch)
        target_prediction = target_aux["soh_pred"]
        if "soh" not in pseudo_target_batch:
            raise ValueError("Pseudo-target batch is missing soh targets.")
        target_prediction_loss = criterion(target_prediction, pseudo_target_batch["soh"])
        target_grads = _gradients(
            target_prediction_loss,
            [(name, parameter) for name, parameter in fast_parameters.items()],
        )
    else:
        # beta=0 is exactly the corresponding ERM gradient path; still report
        # a finite target metric without creating a target gradient graph.
        with torch.no_grad():
            target_aux = _call_model(model, pseudo_target_batch)
            target_prediction_loss = criterion(
                target_aux["soh_pred"], pseudo_target_batch["soh"]
            )
        target_grads = {name: None for name, _ in parameters}

    outer_source_loss = erm_loss + lambda_balance * balance_loss
    source_grads = _gradients(outer_source_loss, parameters)
    for name, parameter in parameters:
        if not parameter.requires_grad:
            parameter.grad = None
            continue
        source = source_grads.get(name)
        target = target_grads.get(name)
        if source is None and target is None:
            parameter.grad = None
            continue
        if source is None:
            source = torch.zeros_like(parameter)
        if target is None:
            target = torch.zeros_like(parameter)
        parameter.grad = source + beta * target

    total_loss = erm_loss + beta * target_prediction_loss + lambda_balance * balance_loss
    return {
        "erm_loss": float(erm_loss.detach().item()),
        "meta_train_loss": float(erm_loss.detach().item()),
        "pseudo_target_loss": float(target_prediction_loss.detach().item()),
        "balance_loss": float(balance_loss.detach().item()),
        "inner_loss": float(inner_loss.detach().item()),
        "total_loss": float(total_loss.detach().item()),
        "inner_learning_rate": alpha,
        "beta": beta,
        "lambda_balance": lambda_balance,
        "inner_steps": 1,
        "fast_parameters_changed": bool(changed),
        "inner_gradient_norm": _gradient_norm(inner_grads),
        "outer_gradient_norm": float(
            torch.sqrt(
                torch.stack(
                    [
                        torch.sum(parameter.grad.detach() ** 2)
                        for _, parameter in parameters
                        if parameter.grad is not None
                    ]
                ).sum()
            ).item()
        )
        if any(parameter.grad is not None for _, parameter in parameters)
        else 0.0,
    }


compute_first_order_mldg_gradients = first_order_mldg_step


__all__ = [
    "MODEL_INPUT_KEYS",
    "compute_first_order_mldg_gradients",
    "first_order_mldg_step",
    "model_inputs_from_batch",
]
