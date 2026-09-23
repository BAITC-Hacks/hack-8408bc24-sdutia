"""Translated option widgets with canonical values independent of wire labels.

Streamlit serializes selectbox/radio selections as their formatted strings.
A new label set therefore needs a new widget identity, while the selected raw
API value must live separately from that widget's serialized state.
"""

from __future__ import annotations

from hashlib import sha256
import json

import streamlit as st


def _state_key(key: str) -> str:
    return f"_choice:{key}"


def get_choice(key: str, default=None):
    return st.session_state.get(_state_key(key), default)


def set_choice(key: str, value) -> None:
    st.session_state[_state_key(key)] = value


def _canonical(value, options, labels, default):
    if value in options:
        return options[options.index(value)]
    if value in labels:
        return options[labels.index(value)]
    return default


def _choice(kind, label, options, *, key, format_func=str, ui=None, **kwargs):
    options = list(options)
    if not options:
        raise ValueError("An option widget requires at least one option")
    labels = [str(format_func(option)) for option in options]
    default = options[0]
    # Migrate a valid old public widget value on an already-open dashboard.
    current = _canonical(get_choice(key, st.session_state.get(key)), options, labels, default)
    set_choice(key, current)
    identity = json.dumps([(repr(option), display) for option, display in zip(options, labels)], ensure_ascii=False)
    widget_key = f"{key}__{sha256(identity.encode('utf-8')).hexdigest()[:12]}"

    def changed():
        set_choice(key, _canonical(st.session_state.get(widget_key), options, labels, current))

    # This also honors programmatic changes, such as Forecast now choosing UTC+5.
    # The callback above records user changes before the next script execution.
    st.session_state[widget_key] = current
    chosen = getattr(ui or st, kind)(
        label, options, index=None, key=widget_key,
        format_func=lambda value: labels[options.index(value)], on_change=changed, **kwargs,
    )
    chosen = _canonical(chosen, options, labels, current)
    set_choice(key, chosen)
    return chosen


def option_selectbox(label, options, *, key, format_func=str, ui=None, **kwargs):
    return _choice("selectbox", label, options, key=key, format_func=format_func, ui=ui, **kwargs)


def option_radio(label, options, *, key, format_func=str, ui=None, **kwargs):
    return _choice("radio", label, options, key=key, format_func=format_func, ui=ui, **kwargs)
