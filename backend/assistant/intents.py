from __future__ import annotations

import re
from dataclasses import dataclass


SUPPORTED_LANGUAGES = {"en", "hi", "hinglish", "mr"}
_PERCENT = re.compile(r"(?P<value>-?\d+(?:\.\d+)?)\s*%")
_PART_ID = re.compile(r"\bP\d{4}\b", re.IGNORECASE)

_INJECTION_MARKERS = (
    "ignore previous",
    "ignore all instructions",
    "system prompt",
    "developer message",
    "reveal secrets",
    "skip confirmation",
    "apply directly",
    "bypass authorization",
    "bypass approval",
    "ignore सुरक्षा",
)

_CATEGORY_SYNONYMS = {
    "Electrical": ("electrical", "विद्युत", "bijli"),
    "Electronics": ("electronics", "इलेक्ट्रॉनिक्स", "इलेक्ट्रॉनिक"),
    "Mechanical": ("mechanical", "यांत्रिक", "यांत्रिक"),
    "Polymer": ("polymer", "पॉलीमर", "प्लास्टिक"),
    "Rubber": ("rubber", "रबर"),
    "Ferrous": ("ferrous", "लौह"),
    "Non-ferrous": ("non-ferrous", "अलौह"),
    "Body": ("body", "बॉडी"),
}


@dataclass(frozen=True)
class Intent:
    name: str
    language: str
    part_id: str | None = None
    category: str | None = None
    percent_change: float | None = None
    clarification_required: bool = False


def detect_language(text: str) -> str:
    devanagari = sum("\u0900" <= char <= "\u097f" for char in text)
    latin = sum(char.isascii() and char.isalpha() for char in text)
    lowered = text.casefold()
    if devanagari:
        if any(word in lowered for word in ("आहे", "करा", "दाखवा", "वाढवा", "कृपया", "समजवा", "शोधा")):
            return "mr"
        if latin:
            return "hinglish"
        return "hi"
    if any(word in lowered for word in ("hai", "dikhao", "batao", "badhao", "karo", "samjhao", "dhundo", "chahiye")):
        return "hinglish"
    return "en"


def extract_intent(text: str) -> Intent:
    normalized = text.casefold()
    language = detect_language(text)

    if any(marker in normalized for marker in _INJECTION_MARKERS):
        return Intent("prompt_injection", language)
    if (
        ("viewer" in normalized or "दर्शक" in normalized)
        and any(word in normalized for word in ("confirm", "approve", "पुष्टि", "मंजूर"))
    ) or "अनुमोदन के बिना" in normalized:
        return Intent("authorization_bypass", language)
    if any(term in normalized for term in ("below moq", "ignore moq", "lead time bypass")):
        return Intent("constraint_violation", language)
    if any(term in normalized for term in ("cancel", "रद्द", "रद्द करें", "रद्द करा")):
        return Intent("cancel", language)
    if re.search(r"\b(revert|undo)\b", normalized) or any(
        term in normalized for term in ("पूर्ववत", "मागे घ्या", "वापस")
    ):
        return Intent("revert", language)
    if any(term in normalized for term in ("unknown part", "not in catalogue", "p9999")):
        match = _PART_ID.search(text)
        return Intent(
            "unknown_part",
            language,
            part_id=match.group(0).upper() if match else None,
        )

    percent_match = _PERCENT.search(normalized)
    if percent_match:
        percentage = float(percent_match.group("value"))
        if percentage < -50 or percentage > 100:
            return Intent("invalid_percentage", language, percent_change=percentage)
        if any(
            term in normalized
            for term in (
                "increase",
                "decrease",
                "adjust",
                "change",
                "बढ़",
                "वाढ",
                "कमी",
                "बदल",
                "badhao",
                "ghatao",
            )
        ):
            return Intent(
                "electrical_example" if "electrical" in normalized else "propose_change",
                language,
                category=_find_category(normalized),
                percent_change=percentage,
                clarification_required=_find_category(normalized) is None,
            )
        return Intent("ambiguous", language, clarification_required=True)

    category = _find_category(normalized)
    if any(
        word in normalized
        for word in ("find", "parts", "category", "खोज", "ढूंढ", "श्रेणी", "शोधा", "भाग")
    ) and category:
        return Intent("find_category", language, category=category)
    if _PART_ID.search(text) and any(
        word in normalized for word in ("part", "stock", "inventory", "इन्वेंटरी", "साठा")
    ):
        return Intent("read_inventory", language, part_id=_PART_ID.search(text).group(0).upper())
    if any(word in normalized for word in ("schedule", "orders", "योजना", "अनुसूची", "शेड्यूल")):
        return Intent("read_schedule", language, category=category)
    if any(word in normalized for word in ("forecast", "demand", "पूर्वानुमान", "मागणी", "डिमांड")):
        return Intent("read_forecast", language, category=category)
    if any(word in normalized for word in ("warning", "risk", "अलर्ट", "चेतावनी", "जोखिम", "इशारा")):
        return Intent("read_warning", language, category=category)
    if any(word in normalized for word in ("kpi", "service level", "fill rate", "performance", "सेवा स्तर", "केपीआई")):
        return Intent("read_kpi", language)
    if any(word in normalized for word in ("inventory", "stock", "on hand", "इन्वेंटरी", "स्टॉक", "साठा")):
        return Intent("read_inventory", language, category=category)
    return Intent("ambiguous", language, clarification_required=True)


def _find_category(text: str) -> str | None:
    for category, terms in _CATEGORY_SYNONYMS.items():
        if any(term.casefold() in text for term in terms):
            return category
    return None


def localized(language: str, key: str, **values: object) -> str:
    messages = {
        "en": {
            "clarify": "Please clarify the part, category, or planning period.",
            "injection": "I can only answer approved read-only questions or create a validated proposal. No changes were made.",
            "permission": "Your role cannot create a schedule proposal. No changes were made.",
            "missing_run": "Provide a completed run ID to retrieve saved planning results.",
            "no_results": "No matching saved result is available for this run.",
            "proposal": "A proposal was created for review. It has not been applied; use the explicit approval action to apply it.",
            "unknown": "That part is not in the imported catalogue. No category or quantity was invented.",
            "cancel": "Cancellation is available through the explicit cancel action; chat text does not change a schedule.",
            "revert": "Reversion requires the explicit revert action and a current approver.",
            "find": "{count} catalogue part(s) match category {category}.",
            "saved": "{count} saved {kind} row(s) are available for this run.",
            "not_understood": "I could not safely resolve that request. Please clarify.",
            "invalid": "The requested percentage is outside the allowed -50% to +100% range.",
            "constraint": "That change conflicts with a hard scheduling constraint and was not proposed.",
        },
        "hi": {
            "clarify": "कृपया भाग, श्रेणी या योजना अवधि स्पष्ट करें।",
            "injection": "मैं केवल स्वीकृत जानकारी दे सकता हूँ या सत्यापित प्रस्ताव बना सकता हूँ। कोई बदलाव नहीं किया गया।",
            "permission": "आपकी भूमिका प्रस्ताव बनाने की अनुमति नहीं देती। कोई बदलाव नहीं किया गया।",
            "missing_run": "सहेजे गए परिणाम देखने के लिए पूर्ण रन आईडी दें।",
            "no_results": "इस रन के लिए कोई मिलान परिणाम उपलब्ध नहीं है।",
            "proposal": "समीक्षा के लिए प्रस्ताव बनाया गया है। इसे लागू नहीं किया गया; स्पष्ट अनुमोदन आवश्यक है।",
            "unknown": "यह भाग आयातित कैटलॉग में नहीं है। कोई श्रेणी या मात्रा नहीं गढ़ी गई।",
            "cancel": "रद्द करने के लिए स्पष्ट cancel कार्रवाई आवश्यक है; चैट पाठ शेड्यूल नहीं बदलता।",
            "revert": "वापस करने के लिए स्पष्ट revert कार्रवाई और वर्तमान approver आवश्यक है।",
            "find": "श्रेणी {category} में {count} कैटलॉग भाग मिले।",
            "saved": "इस रन के लिए {count} सहेजी गई {kind} पंक्तियाँ उपलब्ध हैं।",
            "not_understood": "अनुरोध सुरक्षित रूप से समझ नहीं आया। कृपया स्पष्ट करें।",
            "invalid": "प्रतिशत अनुमत -50% से +100% सीमा के बाहर है।",
            "constraint": "यह बदलाव शेड्यूल की कठोर शर्त से टकराता है; प्रस्ताव नहीं बनाया गया।",
        },
        "hinglish": {
            "clarify": "Part, category ya planning period please clear karo.",
            "injection": "Main sirf approved read-only jawab ya validated proposal de sakta hoon. Koi change nahi hua.",
            "permission": "Aapke role ko proposal banane ki permission nahi hai. Koi change nahi hua.",
            "missing_run": "Saved results ke liye completed run ID do.",
            "no_results": "Is run ke liye matching saved result available nahi hai.",
            "proposal": "Review ke liye proposal bana hai; apply nahi hua. Explicit approval action zaroori hai.",
            "unknown": "Part imported catalogue mein nahi hai. Category ya quantity invent nahi ki.",
            "cancel": "Cancel ke liye explicit cancel action chahiye; chat text schedule change nahi karta.",
            "revert": "Revert ke liye explicit action aur current approver chahiye.",
            "find": "Category {category} mein {count} catalogue part(s) mile.",
            "saved": "Is run ke {kind} ke {count} saved row(s) hain.",
            "not_understood": "Request safely resolve nahi hua. Please clarify karo.",
            "invalid": "Percentage allowed -50% se +100% range ke bahar hai.",
            "constraint": "Yeh change hard scheduling constraint todta hai; proposal nahi bana.",
        },
        "mr": {
            "clarify": "कृपया भाग, श्रेणी किंवा नियोजन कालावधी स्पष्ट करा.",
            "injection": "मी फक्त मंजूर माहिती देऊ शकतो किंवा तपासलेला प्रस्ताव तयार करू शकतो. कोणताही बदल केलेला नाही.",
            "permission": "आपल्या भूमिकेला प्रस्ताव तयार करण्याची परवानगी नाही. बदल केलेला नाही.",
            "missing_run": "साठवलेले परिणाम पाहण्यासाठी पूर्ण रन आयडी द्या.",
            "no_results": "या रनसाठी जुळणारा साठवलेला परिणाम उपलब्ध नाही.",
            "proposal": "पुनरावलोकनासाठी प्रस्ताव तयार केला आहे; तो लागू केलेला नाही. स्पष्ट मंजुरी आवश्यक आहे.",
            "unknown": "हा भाग आयात केलेल्या कॅटलॉगमध्ये नाही. श्रेणी किंवा प्रमाण बनवलेले नाही.",
            "cancel": "रद्द करण्यासाठी स्पष्ट cancel कृती आवश्यक आहे; चॅट मजकूर वेळापत्रक बदलत नाही.",
            "revert": "परत करण्यासाठी स्पष्ट revert कृती आणि सध्याचा approver आवश्यक आहे.",
            "find": "श्रेणी {category} मध्ये {count} कॅटलॉग भाग सापडले.",
            "saved": "या रनसाठी {count} साठवलेल्या {kind} नोंदी उपलब्ध आहेत.",
            "not_understood": "विनंती सुरक्षितपणे समजली नाही. कृपया स्पष्ट करा.",
            "invalid": "टक्केवारी अनुमत -50% ते +100% मर्यादेबाहेर आहे.",
            "constraint": "हा बदल कठोर नियोजन अटीशी विसंगत आहे; प्रस्ताव केला नाही.",
        },
    }
    return messages.get(language, messages["en"]).get(key, messages["en"]["not_understood"]).format(**values)
