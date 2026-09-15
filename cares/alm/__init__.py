from .base import (
    MAX_NEW_TOKENS,
    AudioChat,
    AudioReply,
    ensure_libstdcxx_on_path,
    preload_libstdcxx,
    read_reply,
    strip_thinking,
)
from .cascade import CascadeChat
from .fake import FakeAudioChat
from .flamingo import FlamingoChat
from .kimi import KimiAudioChat
from .midasheng import MiDashengLMChat
from .mimo import MimoAudioChat
from .moss import MossAudioChat
from .qwen_omni import QwenOmniChat

BACKENDS = {
    FakeAudioChat.name: FakeAudioChat,
    FlamingoChat.name: FlamingoChat,
    KimiAudioChat.name: KimiAudioChat,
    QwenOmniChat.name: QwenOmniChat,
    MossAudioChat.name: MossAudioChat,
    MimoAudioChat.name: MimoAudioChat,
    MiDashengLMChat.name: MiDashengLMChat,
    CascadeChat.name: CascadeChat,
}

__all__ = ["BACKENDS", "MAX_NEW_TOKENS", "AudioChat", "AudioReply", "CascadeChat",
           "ensure_libstdcxx_on_path",
           "FakeAudioChat", "FlamingoChat", "KimiAudioChat", "MiDashengLMChat",
           "MimoAudioChat",
           "MossAudioChat",
           "QwenOmniChat", "preload_libstdcxx", "read_reply", "strip_thinking"]
