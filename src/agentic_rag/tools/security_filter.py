import os
import re
from typing import Dict, List, Tuple, Optional
import logging


try:
    from guardrails.validators import ProfanityFree  
except Exception:  
    ProfanityFree = None  


try:
    from openai import OpenAI  
except Exception:  
    OpenAI = None  

class SecurityFilter:
    """
    Security filter for PDPA Assistant to prevent inappropriate content and restrict topics to PDPA only.
    """
    
    def __init__(self):
   
        self.llama_base_url = os.getenv("LLM_BASE_URL", os.getenv("LLAMA_CPP_BASE_URL", "https://api.opentyphoon.ai/v1"))
        self.llama_model = os.getenv("LLM_MODEL", os.getenv("LLAMA_CPP_MODEL", "typhoon-v2.5-30b-a3b-instruct"))
        api_key = os.getenv("LLM_API_KEY") or os.getenv("TYPHOON_API_KEY") or os.getenv("OPENAI_API_KEY") or "not-needed"
        self._openai_client = None
        if OpenAI is not None:
            try:
                self._openai_client = OpenAI(base_url=self.llama_base_url, api_key=api_key)
            except Exception:
                self._openai_client = None

        self._profanity_validator = None
        if ProfanityFree is not None:
            try:
                self._profanity_validator = ProfanityFree()
            except Exception:
                self._profanity_validator = None

        self.thai_inappropriate_terms = [
            "หี", "จิ๋ม", "จู๋", "ไข่", "หำ", "แตด", "หัวนม", "เย็ด",
            "ปี้", "เสียบ", "เสียว", "เงี่ยน", "ข่มขืน", "รุมโทรม",
            "น้ำแตก", "แตกใน", "สวิงกิ้ง", "สวิ้งกิ้ง", "ดูดปาก", "อมควย", "เลียหี",
            "ตูด", "ส้นตีน", "ตีน", "ตอแหล", "พ่อง", "พ่อมึง", "แม่มึง", "พ่อมึงตาย",
            "แม่มึงตาย", "เหี้ย", "เฮี้ย", "เห้", "ห่า", "สัด", "สัส", "เชี่ย",
            "เชี้ย", "เชี่ยเอ้ย", "เชี้ยเอ้ย", "แม่ง", "มึง", "กู", "ไอสัด", "ไอ้สัด",
            "ไอสัส", "ไอ้สัส", "ไอ้เหี้ย", "อีเหี้ย", "เหี้ยมาก", "เหี้ยสุด", "ไอ้ห่า",
            "ไอสาด", "ไอสาดด", "สัตว์", "ชาติหมา", "ไอ้ชาติหมา", "ไอ้เวร", "ไอ้เวรตะไล",
            "ไอ้ควาย", "ควาย", "ไอ้โง่", "โง่เง่า", "โง่ควาย", "ปัญญาอ่อน", "ไร้สมอง",
            "สมองหมา", "ไอ้บ้า", "สวะ", "เฮงซวย", "ระยำ", "ไอ้ระยำ", "สถุน", "ไอ้สถุน",
            "ต่ำตม", "อัปรีย์", "จัญไร", "กระหรี่", "กะหรี่", "กระรี่", "กะรี่", "อีตัว", "อีแพศยา", "แพศยา",
            "อีดอก", "ดอกทอง", "หน้าควย", "หน้าหี", "ไอ้หน้าควย", "ไอ้หน้าหี",
            "ชิบหาย", "ชิบหายวายวอด", "ไอ้สารเลว", "สารเลว", "เลว", "สถุนมาก",
            "หน้าด้าน", "หน้าหนา", "หน้าตัวเมีย", "กาก", "ขยะสังคม", "มะเร็งสังคม",
            "ควยใหญ่", "ควยยาว", "ควยเด็ก", "หีบาน", "หีเด็ก", "หีเน่า", "จิ๋มเด็ก",
            "แตกคาปาก", "แตกคา", "แตกใส่หน้า", "แตกใส่ปาก", "เสร็จใน", "ขย่ม", "ขยี้",
            "ขึ้นคร่อม", "กระแทก", "เอากัน", "เอาอย่างว่า", "เอาแรงๆ", "เอาแรงแรง",
            "เลีย", "ดูด",  "แหย่", "เสียบเข้า", "สอดเข้า",
            "ต่ำตม", "อัปรีย์", "จัญไร", "กระหรี่", "กะหรี่", "อีตัว", "อีแพศยา", "แพศยา",
            "อีดอก", "ดอกทอง", "หน้าควย", "หน้าหี", "ไอ้หน้าควย", "ไอ้หน้าหี"
        ]


        thai_variant_patterns = [
            r"ค\s*วย(?:ย+)?",                    
            r"(?:ไอ้|อี)?\s*เหี้ย(?:ย+)?",   
            r"(?:ไอ้|อี)?\s*สัด",          
            r"(?:ไอ้|อี)?\s*สัส",          
            r"(?:ไอ้|อี)?\s*เวร(?:ตะไล)?", 
            r"(?:ไอ้|อี)?\s*ควาย",        
            r"(?:ไอ้|อี)?\s*ระยำ",        
            r"(?:ไอ้|อี)?\s*สถุน",        
            r"ชาติ\s*หมา",                 
            r"หน้า\s*ควย",                 
            r"หน้า\s*หี"                   
        ]


        thai_literals_pattern = "|".join(map(re.escape, self.thai_inappropriate_terms))
        thai_pattern = rf"({thai_literals_pattern}|{'|'.join(thai_variant_patterns)})"

 
        self.english_inappropriate_terms = [
            "fuck", "motherfucker", "mf", "shit", "bullshit", "bs", "bitch",
            "slut", "whore", "cunt", "pussy", "dick", "cock", "asshole",
            "ass", "bastard", "son of a bitch", "sob", "bloody hell",
            "goddamn", "god damn", "damn", "prick", "wanker", "twat"
        ]
        english_variant_patterns = [
            r"f\*+k", r"fxxk", r"sh\*t", r"b!tch"
        ]
        english_literals_pattern = "|".join(map(re.escape, self.english_inappropriate_terms))
        english_pattern = rf"\\b({english_literals_pattern}|{'|'.join(english_variant_patterns)})\\b"


        self.security_avoid_terms = [
            "crack", "cracker", "cracking", "exploit", "vulnerability", "intrusion",
            "malware", "virus", "trojan", "ransomware", "phishing", "spam", "ddos",
            "sql injection", "xss", "csrf", "buffer overflow", "privilege escalation",
            "rootkit", "backdoor", "keylogger", "spyware", "adware", "botnet", "worm",
            "logic bomb", "time bomb", "easter egg", "trapdoor", "trap door", "trap-door",
            "trap_door"
        ]
        security_pattern = rf"\\b({'|'.join(map(re.escape, self.security_avoid_terms))})\\b"

        self.violence_terms_th = ["ฆ่า", "ฆาตกรรม", "ฆาตกร"]

        violence_th_pattern = rf"(?<![ก-๙])({'|'.join(map(re.escape, self.violence_terms_th))})(?![ก-๙])"
        self.violence_terms_en = [
            "kill", "murder", "murderer", "assassinate", "assassin", "execute",
            "execution", "suicide", "suicidal", "terrorism", "terrorist", "bomb",
            "explosion", "gun", "weapon", "violence", "violent", "attack", "assault",
            "threat", "threatening"
        ]
        violence_en_pattern = rf"\\b({'|'.join(map(re.escape, self.violence_terms_en))})\\b"


        self.drug_terms_th = ["ยาเสพติด", "ยาเสพ"]

        drug_th_pattern = rf"(?<![ก-๙])({'|'.join(map(re.escape, self.drug_terms_th))})(?![ก-๙])"
        self.drug_terms_en = [
            "drug", "drugs", "heroin", "cocaine", "marijuana", "weed", "meth",
            "amphetamine", "ecstasy", "lsd", "pills", "overdose", "addiction",
            "addict", "dealer", "trafficking", "smuggling"
        ]
        drug_en_pattern = rf"\\b({'|'.join(map(re.escape, self.drug_terms_en))})\\b"

        self.inappropriate_patterns = [
            thai_pattern,
            english_pattern,
            security_pattern,
            violence_th_pattern,
            violence_en_pattern,
            drug_th_pattern,
            drug_en_pattern,
        ]
        

        self.pdpa_keywords = []
        

        self.inappropriate_regex = re.compile('|'.join(self.inappropriate_patterns), re.IGNORECASE)

        self.injection_phrases = [
            r"ignore (all|any) (previous|prior) (instructions|messages)",
            r"disregard (the )?(rules|system|guardrails)",
            r"act as (?:an?|the) (?:admin|developer|system)",
            r"reveal (?:your )?(?:system|hidden) prompt",
            r"jailbreak|do-anything-now|DAN",
            r"override safety|bypass safety|disable safety",
            r"เพิกเฉยคำสั่งก่อนหน้า|ละเลยกฎ|แสดง system prompt|ปิดการทำงานความปลอดภัย|ข้ามข้อจำกัด",
        ]
        self.injection_regex = re.compile('|'.join(self.injection_phrases), re.IGNORECASE)


        self.email_regex = re.compile(r"[a-zA-Z0-9_.+\-]+@[a-zA-Z0-9\-]+\.[a-zA-Z0-9\-.]+")
        self.phone_regex = re.compile(r"(?:\+?66|0)[\s\-]?(?:\d[\s\-]?){8,10}")

        self.thai_id_regex = re.compile(r"\b\d{1}-?\d{4}-?\d{5}-?\d{2}-?\d{1}\b")
        

        self.logger = logging.getLogger(__name__)
    
    def check_content_safety(self, text: str) -> Tuple[bool, List[str]]:
        """
        Content safety check using Guardrails profanity validator if available,
        with fallback to regex-based detection of severe profanity/violence, etc.
        
        Args:
            text: The text to check
            
        Returns:
            Tuple of (is_safe, list_of_violations)
        """
        if not text:
            return True, []
        
        violations = []
        text_lower = text.lower()


        if self._profanity_validator is not None:
            try:

                is_clean = True
                message = None
                if hasattr(self._profanity_validator, "validate"):
                    result = self._profanity_validator.validate(text_lower)  # type: ignore

                    if isinstance(result, tuple) and len(result) >= 2:

                        _, err = result[0], result[1]
                        is_clean = err is None
                        message = None if err is None else str(err)
                    elif isinstance(result, dict):

                        is_clean = bool(result.get("is_valid", True))
                        message = result.get("error")
                    else:

                        is_clean = True
                else:

                    _ = self._profanity_validator(text_lower) 

                if not is_clean:
                    violations.append("ตรวจพบคำหยาบจาก Guardrails Profanity validator" if not message else message)
            except Exception:
                pass
        
        matches = self.inappropriate_regex.findall(text_lower)
        if matches:
            flat_matches = []
            for match in matches:
                if isinstance(match, tuple):
                    flat_matches.extend([m for m in match if m])
                else:
                    flat_matches.append(match)
            if flat_matches:
                violations.append(f"พบคำที่ไม่เหมาะสม: {', '.join(set(flat_matches))}")
        
        extreme_security_terms = [
            "create malware", "create virus", "create trojan", "create ransomware",
            "make malware", "make virus", "make trojan", "make ransomware",
            "build malware", "build virus", "build trojan", "build ransomware",
            "develop malware", "develop virus", "develop trojan", "develop ransomware"
        ]
        
        found_extreme_terms = []
        for term in extreme_security_terms:
            if term.lower() in text_lower:
                found_extreme_terms.append(term)
        
        if found_extreme_terms:
            violations.append(f"พบคำที่เกี่ยวข้องกับการสร้างมัลแวร์: {', '.join(found_extreme_terms)}")
        
        is_safe = len(violations) == 0
        
        if not is_safe:
            self.logger.warning(f"Content safety violation detected: {violations}")
        
        return is_safe, violations

    def _ai_check_pdpa_related(self, text: str) -> Tuple[bool, str]:
        """
        Determine PDPA-relatedness using ONLY the AI judgment (per request).
        If the AI is not available, treat as not related and ask user to ask about PDPA.
        Returns (is_related, reason_text).
        """
        if not text or self._openai_client is None:
            return False, "AI unavailable for PDPA check"

        try:
            system = (
                "คุณเป็นผู้เชี่ยวชาญกฎหมาย PDPA ไทย ให้ตัดสินว่า 'ข้อความเกี่ยวข้องกับ PDPA หรือไม่' จากสาระทางกฎหมาย ไม่ใช่แค่มีคำว่า PDPA\n"
                "- เกี่ยวข้อง: สิทธิของเจ้าของข้อมูล (ขอเข้าถึง/ลบ/ถอนความยินยอม/คัดค้าน), หน้าที่ผู้ควบคุม/ผู้ประมวลผล, ฐานกฎหมาย (ยินยอม/สัญญา/หน้าที่กม./ประโยชน์ชอบธรรม/ภารกิจรัฐ/สาธารณประโยชน์), การเก็บ-ใช้-เปิดเผย-โอนข้ามแดน, การแจ้งเหตุละเมิด, บันทึกรายการประมวลผล, DPO, DPIA, สัญญา data processing, การถ่ายภาพ/วิดีโอที่เห็นบุคคลแล้วเผยแพร่ออนไลน์, CCTV/ไบโอเมตริกซ์/คุกกี้/การตลาดตรงที่แตะข้อมูลส่วนบุคคล\n"
                "- ไม่เกี่ยวข้อง: แค่เอ่ยชื่อ PDPA หรือใช้ PDPA เป็นชื่อสินค้า/บริการ/ร้านค้า/กระเป๋า หรือประเด็นที่ไม่ใช่กฎหมายคุ้มครองข้อมูลส่วนบุคคล\n"
                "- ตัวอย่างเกี่ยวข้อง: 'ถ่ายรูปติดคนอื่นแล้วโพสต์ได้ไหม', 'ต้องขอความยินยอมไหม', 'สิทธิขอเข้าถึงข้อมูล', 'ฐานกฎหมายใดใช้ได้', 'ต้องแจ้งเหตุละเมิดเมื่อใด', 'โอนข้อมูลออกนอกประเทศต้องมีอะไร'\n"
                "- ตัวอย่างไม่เกี่ยวข้อง: 'กระเป๋า PDPA ราคาเท่าไหร่', 'PDPA คือชื่อร้านอาหาร', 'พูดถึง PDPA แต่ถามเรื่องเกมหรือหุ้น'\n"
                "ตอบเหตุผลให้กระชับ รูปแบบ: 'เหตุผลที่เกี่ยวข้อง: ' หรือ 'เหตุผลที่ไม่เกี่ยวข้อง: ' ให้อธิบายว่าทำไมถึงเกี่ยวข้องหรือไม่เกี่ยวข้อง ให้คิดใหม่ทุกครั้ง"
            )
            prompt = f"ข้อความ: {text}\nโปรดตอบรูปแบบสั้น ๆ ตามที่กำหนดเท่านั้น"
            resp = self._openai_client.chat.completions.create(
                model=self.llama_model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                max_tokens=120,
                temperature=0
            )
            content = (resp.choices[0].message.content or "").strip()
            norm = content.replace("\u200b", "").strip().lower()

            if norm.startswith("y:") or norm.startswith("เหตุผลที่เกี่ยวข้อง"):
                return True, content
            if norm.startswith("n:") or norm.startswith("เหตุผลที่ไม่เกี่ยวข้อง"):
                return False, content

            return False, content or "ไม่เข้าใจรูปแบบคำตอบจาก AI"
        except Exception as e:
            return False, "AI error during PDPA check"

    def _is_severe_profanity(self, text: str, violations: List[str]) -> bool:
        """
        Check if the profanity is severe enough to block the question.
        
        Args:
            text: The text to check
            violations: List of safety violations found
            
        Returns:
            True if the profanity is severe, False otherwise
        """
        if not violations:
            return False

        severe_patterns = [

            "เย็ด", "ปี้", "สอด", "เสียบ", "เสียว", "เงี่ยน", "ข่มขืน", "รุมโทรม",
            "น้ำแตก", "แตกใน", "สวิงกิ้ง", "สวิ้งกิ้ง", "ดูดปาก", "อมควย", "เลียหี",
            "ควยใหญ่", "ควยยาว", "ควยเด็ก", "หีบาน", "หีเด็ก", "หีเน่า", "จิ๋มเด็ก",
            "แตกคาปาก", "แตกคา", "แตกใส่หน้า", "แตกใส่ปาก", "เสร็จใน", "ขย่ม", "ขยี้",
            "ขึ้นคร่อม", "กระแทก", "เอากัน", "เอาอย่างว่า", "เอาแรงๆ", "เอาแรงแรง",
            "เลีย", "ดูด", "อม", "ยัด", "แหย่", "เสียบเข้า", "สอดเข้า",
            

            "พ่อมึงตาย", "แม่มึงตาย", "ชิบหาย", "ชิบหายวายวอด", "ไอ้สารเลว", "สารเลว",
            "หน้าด้าน", "หน้าหนา", "หน้าตัวเมีย", "กาก", "ขยะสังคม", "มะเร็งสังคม",
            "กระหรี่", "กะหรี่", "กระรี่", "กะรี่", "อีตัว", "อีแพศยา", "แพศยา",
            "อีดอก", "ดอกทอง", "หน้าควย", "หน้าหี", "ไอ้หน้าควย", "ไอ้หน้าหี",
            
   
            "fuck", "motherfucker", "cunt", "pussy", "dick", "cock", "asshole",
            "bitch", "slut", "whore", "bastard", "son of a bitch"
        ]
        
        text_lower = text.lower()
        
        for pattern in severe_patterns:
            if pattern.lower() in text_lower:
                return True
        
        mild_profanities = ["มึง", "กู", "เหี้ย", "เฮี้ย", "เห้", "ห่า", "สัด", "สัส", "เชี่ย", "เชี้ย", "แม่ง"]
        mild_count = sum(1 for word in mild_profanities if word in text_lower)
        
        if mild_count > 2:
            return True
        
        return False

    def detect_prompt_injection(self, text: str) -> List[str]:
        """
        Detect likely prompt-injection attempts using heuristic patterns.
        Returns list of matched phrases (empty if none).
        """
        if not text:
            return []
        matches = self.injection_regex.findall(text)

        flat = []
        for m in matches:
            if isinstance(m, tuple):
                flat.extend([x for x in m if x])
            else:
                flat.append(m)
        return list(set(flat))

    def sanitize_pii(self, text: str) -> str:
        """
        Redact common PII patterns from text.
        """
        if not text:
            return text
        redacted = self.email_regex.sub('[REDACTED_EMAIL]', text)
        redacted = self.phone_regex.sub('[REDACTED_PHONE]', redacted)
        redacted = self.thai_id_regex.sub('[REDACTED_THAI_ID]', redacted)
        return redacted
    
    def check_topic_restriction(self, text: str) -> Tuple[bool, List[str]]:
        """
        Check if the text is related to PDPA topics only.
        
        Args:
            text: The text to check
            
        Returns:
            Tuple of (is_pdpa_related, list_of_reasons)
        """
        if not text:
            return False, ["ข้อความว่างเปล่า"]
        
        text_lower = text.lower()
        

        pdpa_matches = []
        

        legal_privacy_terms = [
            "กฎหมาย", "กฎหมายคุ้มครอง", "กฎหมายข้อมูล", "กฎหมายส่วนบุคคล",
            "privacy", "legal", "law", "regulation", "compliance", "governance",
            "ข้อมูล", "ข้อมูลส่วนบุคคล", "ข้อมูลส่วนตัว", "ข้อมูลส่วนบุคคล",
            "data", "personal", "private", "confidential", "sensitive",

            "ความปลอดภัย", "security", "protection", "คุ้มครอง", "protect",
            "ความเป็นส่วนตัว", "privacy", "ส่วนตัว", "private",
            "ข้อมูล", "information", "data", "สารสนเทศ",
            "กฎหมาย", "law", "regulation", "กฎระเบียบ",
            "การจัดการ", "management", "การบริหาร", "administration"
        ]
        
        legal_matches = []
        for term in legal_privacy_terms:
            if term.lower() in text_lower:
                legal_matches.append(term)
        
        if legal_matches:
            return True, [f"เกี่ยวข้องกับกฎหมาย/ความเป็นส่วนตัว: {', '.join(legal_matches[:3])}"]
        
        return False, ["ไม่เกี่ยวข้องกับ PDPA หรือกฎหมายคุ้มครองข้อมูลส่วนบุคคล"]
    
    def filter_user_input(self, user_input: str) -> Dict[str, any]:
        """
        [CORRECTED LOGIC]
        ตรวจสอบ Input จากผู้ใช้ตามลำดับความสำคัญ:
        1. ตรวจจับการโจมตี (Injection) -> บล็อกทันที
        2. ตรวจจับเนื้อหาไม่เหมาะสม (Profanity) -> บล็อกทันที
        3. ตรวจสอบว่าหัวข้อเกี่ยวกับ PDPA หรือไม่ -> บล็อกถ้าไม่เกี่ยว
        """

        result = {
            "is_safe": True,
            "is_pdpa_related": True,
            "violations": [],
            "reasons": [],
            "filtered_text": user_input,
            "should_respond": True,
            "response_message": ""
        }


      
        injection_hits = self.detect_prompt_injection(user_input)
        if injection_hits:
            result["should_respond"] = False
            result["response_message"] = "ตรวจพบความพยายามในการโจมตีระบบ"
            result["violations"].append(f"Prompt-injection attempt: {', '.join(injection_hits)}")
            return result

        
        is_safe, safety_violations = self.check_content_safety(user_input)
        if not is_safe:
            # บล็อกคำหยาบทันที ไม่ว่าจะเกี่ยวข้องกับ PDPA หรือไม่
            result["should_respond"] = False
            result["response_message"] = "🔴 [Guardrail] บล็อกคำถามเนื่องจากพบคำหยาบ/ไม่เหมาะสม"
            result["violations"].extend(safety_violations)
            return result


        if os.getenv("DISABLE_PDPA_CHECK", "0").lower() in ("1", "true", "yes", "y"):
            return result

        is_pdpa_related, reason_text = self._ai_check_pdpa_related(user_input)
        if not is_pdpa_related:
            result["should_respond"] = False
        
            result["response_message"] = (
                "🔴 หัวข้อนี้ไม่เกี่ยวข้องกับ PDPA (พ.ร.บ.คุ้มครองข้อมูลส่วนบุคคล)\n"
                f"- {reason_text}\n"
            )
            if reason_text:
                result["reasons"].append(reason_text)
            return result


        return result
    
    def filter_ai_response(self, ai_response: str) -> Dict[str, any]:
        """
        Filter AI-generated responses for safety.
        
        Args:
            ai_response: The AI's response text
            
        Returns:
            Dictionary with filtering results
        """
        result = {
            "is_safe": True,
            "violations": [],
            "filtered_text": ai_response,
            "should_display": True,
            "replacement_message": ""
        }

        ai_response = self.sanitize_pii(ai_response)


        is_safe, safety_violations = self.check_content_safety(ai_response)
        result["is_safe"] = is_safe
        result["violations"].extend(safety_violations)
        
        if not is_safe:
            result["should_display"] = False
            result["replacement_message"] = (
                "ขออภัย ฉันไม่สามารถแสดงคำตอบที่มีเนื้อหาไม่เหมาะสมได้ "
                "โปรดลองถามคำถามใหม่ที่เกี่ยวข้องกับ PDPA"
            )
        

        injection_hits = self.detect_prompt_injection(ai_response)
        if injection_hits:
            result["violations"].append(
                f"ตรวจพบข้อความที่อาจเป็น prompt-injection: {', '.join(injection_hits[:3])}"
            )
        return result
    
    def sanitize_text(self, text: str) -> str:
        """
        Sanitize text by removing or replacing inappropriate content.
        """
        if not text:
            return text

        sanitized = self.inappropriate_regex.sub('[REDACTED]', text)
        sanitized = self.sanitize_pii(sanitized)
        return sanitized
