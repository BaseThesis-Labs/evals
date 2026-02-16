# """
# Task Success Rate (TSR) & Completion Metrics
# Implements PARADISE framework, slot-filling tracking, and containment metrics
# Designed for goal-oriented dialogues like travel booking agents
# """

# import json
# import re
# import numpy as np
# from typing import List, Dict, Set, Optional, Tuple, Any
# from dataclasses import dataclass, asdict, field
# from enum import Enum
# import warnings
# warnings.filterwarnings('ignore')


# class TaskStatus(Enum):
#     """Task completion status"""
#     COMPLETED = "completed"
#     PARTIAL = "partial"
#     FAILED = "failed"
#     ABANDONED = "abandoned"


# class SlotStatus(Enum):
#     """Individual slot status"""
#     FILLED = "filled"
#     PARTIALLY_FILLED = "partially_filled"
#     MISSING = "missing"
#     INCORRECT = "incorrect"


# @dataclass
# class Slot:
#     """Represents a required information slot"""
#     name: str
#     required: bool
#     value: Optional[str] = None
#     status: SlotStatus = SlotStatus.MISSING
#     confidence: float = 0.0
#     turn_filled: int = -1  # Which turn it was filled
#     changed: bool = False  # Was it changed after initial fill

#     @property
#     def filled(self) -> bool:
#         """Convenience property to check if slot is filled"""
#         return self.status == SlotStatus.FILLED

#     @filled.setter
#     def filled(self, value: bool):
#         """Set slot status based on filled boolean"""
#         if value:
#             self.status = SlotStatus.FILLED
#         else:
#             self.status = SlotStatus.MISSING


# @dataclass
# class TravelAgentSlots:
#     """Specific slots for travel agent task"""
#     # Required slots
#     destination: Slot = field(default_factory=lambda: Slot("destination", required=True))
#     checkin_date: Slot = field(default_factory=lambda: Slot("checkin_date", required=True))
#     checkout_date: Slot = field(default_factory=lambda: Slot("checkout_date", required=True))
#     num_guests: Slot = field(default_factory=lambda: Slot("num_guests", required=True))
#     budget_min: Slot = field(default_factory=lambda: Slot("budget_min", required=True))
#     budget_max: Slot = field(default_factory=lambda: Slot("budget_max", required=True))

#     # Optional slots
#     accommodation_type: Slot = field(default_factory=lambda: Slot("accommodation_type", required=False))
#     amenities: Slot = field(default_factory=lambda: Slot("amenities", required=False))
#     neighborhood: Slot = field(default_factory=lambda: Slot("neighborhood", required=False))
#     special_requests: Slot = field(default_factory=lambda: Slot("special_requests", required=False))

#     def get_all_slots(self) -> List[Slot]:
#         """Get all slot objects"""
#         return [
#             self.destination, self.checkin_date, self.checkout_date,
#             self.num_guests, self.budget_min, self.budget_max,
#             self.accommodation_type, self.amenities,
#             self.neighborhood, self.special_requests
#         ]

#     def get_required_slots(self) -> List[Slot]:
#         """Get only required slots"""
#         return [s for s in self.get_all_slots() if s.required]

#     def get_optional_slots(self) -> List[Slot]:
#         """Get only optional slots"""
#         return [s for s in self.get_all_slots() if not s.required]


# @dataclass
# class TaskCompletionMetrics:
#     """Complete task success metrics"""
#     # Task Success Rate (TSR)
#     tsr: float  # Overall TSR percentage
#     total_tasks: int
#     completed_tasks: int
#     partial_tasks: int
#     failed_tasks: int
#     abandoned_tasks: int

#     # Slot filling metrics
#     required_slots_filled: int
#     required_slots_total: int
#     optional_slots_filled: int
#     optional_slots_total: int
#     slot_filling_rate: float  # Percentage of required slots filled

#     # PARADISE framework metrics
#     kappa_statistic: float  # Information exchange quality
#     dialogue_success: float  # Binary or scaled success
#     dialogue_cost: float  # Number of turns / efficiency
#     user_satisfaction_estimate: float  # Estimated from PARADISE model

#     # Containment metrics
#     first_call_resolution: float  # Percentage resolved in first attempt
#     containment_rate: float  # Percentage handled without escalation
#     false_containment_rate: float  # Abandoned without completion

#     # Efficiency metrics
#     avg_turns_to_completion: float
#     avg_slots_per_turn: float
#     slot_correction_rate: float  # How often slots needed correction

#     # Detailed breakdown
#     slot_details: Dict[str, Dict[str, Any]]
#     task_breakdown: Dict[str, int]


# @dataclass
# class DialogueTurn:
#     """Represents a single dialogue turn"""
#     turn_id: int
#     speaker: str  # 'user' or 'agent'
#     text: str
#     extracted_slots: Dict[str, str]
#     timestamp: float = 0.0


# @dataclass
# class TaskSession:
#     """Complete task session"""
#     session_id: str
#     turns: List[DialogueTurn]
#     slots: TravelAgentSlots
#     status: TaskStatus
#     escalated: bool = False
#     abandoned: bool = False
#     user_satisfied: Optional[bool] = None
#     completion_time: float = 0.0


# class TaskCompletionEvaluator:
#     """
#     Evaluates task completion for goal-oriented dialogues

#     Tracks:
#     - Task Success Rate (TSR)
#     - Slot filling completion
#     - PARADISE framework metrics
#     - Containment and escalation
#     - False containment detection
#     """

#     def __init__(self,
#                  required_slots: Optional[List[str]] = None,
#                  optional_slots: Optional[List[str]] = None):
#         """
#         Initialize evaluator

#         Args:
#             required_slots: List of required slot names
#             optional_slots: List of optional slot names
#         """
#         # Default travel agent slots
#         self.default_required = [
#             'destination', 'checkin_date', 'checkout_date',
#             'num_guests', 'budget_min', 'budget_max'
#         ]

#         self.default_optional = [
#             'accommodation_type', 'amenities',
#             'neighborhood', 'special_requests'
#         ]

#         self.required_slots = required_slots or self.default_required
#         self.optional_slots = optional_slots or self.default_optional

#         # Extraction patterns for travel agent
#         self._init_extraction_patterns()

#     def _init_extraction_patterns(self):
#         """Initialize regex patterns for slot extraction"""
#         self.patterns = {
#             'destination': [
#                 r'(?:going to|travel(?:ing)? to|visit(?:ing)?|in)\s+([A-Z][a-zA-Z\s]+?)(?:\s+in|\s+from|\.|,|$)',
#                 r'([A-Z][a-zA-Z\s]+?)(?:\s+from|\s+in\s+\w+)|$'
#             ],
#             'checkin_date': [
#                 r'(?:from|check-?in|arrive|arriving|start(?:ing)?)\s+(?:on\s+)?([A-Za-z]+\s+\d{1,2}(?:st|nd|rd|th)?)',
#                 r'(\w+\s+\d{1,2}(?:st|nd|rd|th)?)\s+to',
#                 r'(\d{1,2}/\d{1,2}(?:/\d{2,4})?)'
#             ],
#             'checkout_date': [
#                 r'(?:to|until|check-?out|leave|leaving|end(?:ing)?)\s+(?:on\s+)?([A-Za-z]+\s+\d{1,2}(?:st|nd|rd|th)?)',
#                 r'to\s+(\w+\s+\d{1,2}(?:st|nd|rd|th)?)',
#                 r'(?:to|until)\s+(\d{1,2}/\d{1,2}(?:/\d{2,4})?)'
#             ],
#             'num_guests': [
#                 r'(\d+)\s+(?:of\s+us|people|guests|adults|travelers)',
#                 r'(?:for|party of)\s+(\d+)',
#                 r'(\w+)\s+(?:of\s+us|people|guests)'  # "three of us"
#             ],
#             'budget_min': [
#                 r'\$?(\d+)\s*(?:to|-)\s*\$?\d+',
#                 r'around\s+\$?(\d+)',
#                 r'budget.*?\$?(\d+)'
#             ],
#             'budget_max': [
#                 r'\$?\d+\s*(?:to|-)\s*\$?(\d+)',
#                 r'up to\s+\$?(\d+)',
#                 r'stretch.*?\$?(\d+)'
#             ],
#             'accommodation_type': [
#                 r'(hotel|airbnb|hostel|resort|apartment|villa|b&b|motel)',
#                 r'(?:looking for|prefer|want).*?(hotel|airbnb|apartment)'
#             ],
#             'amenities': [
#                 r'(pool|gym|wifi|parking|kitchen|breakfast|spa|beach)',
#                 r'(?:with|need|want).*?(pool|gym|wifi|parking)'
#             ],
#             'neighborhood': [
#                 r'(?:in|near|around)\s+([A-Z][a-zA-Z\s]+?)(?:\s+area|\s+district|,|\.|$)',
#                 r'(?:soma|downtown|financial district|mission|castro|haight)'  # SF specific
#             ]
#         }

#     def extract_slots_from_text(self, text: str) -> Dict[str, str]:
#         """
#         Extract slot values from text using patterns

#         Args:
#             text: Input text

#         Returns:
#             Dictionary of extracted slot values
#         """
#         extracted = {}
#         text_lower = text.lower()

#         for slot_name, patterns in self.patterns.items():
#             for pattern in patterns:
#                 match = re.search(pattern, text, re.IGNORECASE)
#                 if match:
#                     value = match.group(1).strip()
#                     extracted[slot_name] = value
#                     break

#         return extracted

#     def evaluate_transcript(self,
#                            transcript: str,
#                            task_successful: bool = True,
#                            user_abandoned: bool = False) -> Tuple[TravelAgentSlots, Dict[str, Any]]:
#         """
#         Simple evaluation from just a transcript string

#         Args:
#             transcript: Full dialogue transcript
#             task_successful: Whether task was completed
#             user_abandoned: Whether user abandoned

#         Returns:
#             Tuple of (filled slots, metrics dict)
#         """
#         # Extract all slots from transcript
#         extracted = self.extract_slots_from_text(transcript)

#         # Create slots object
#         slots = TravelAgentSlots()

#         # Fill slots
#         for slot_name, value in extracted.items():
#             slot = getattr(slots, slot_name, None)
#             if slot:
#                 slot.value = value
#                 slot.status = SlotStatus.FILLED

#         # Calculate completion metrics
#         required = slots.get_required_slots()
#         filled_required = [s for s in required if s.status == SlotStatus.FILLED]
#         required_completion = len(filled_required) / len(required) if required else 0.0

#         optional = slots.get_optional_slots()
#         filled_optional = [s for s in optional if s.status == SlotStatus.FILLED]
#         optional_completion = len(filled_optional) / len(optional) if optional else 0.0

#         # Determine status
#         if len(filled_required) == len(required) and task_successful:
#             status = TaskStatus.COMPLETED
#         elif len(filled_required) > 0:
#             status = TaskStatus.PARTIAL
#         elif user_abandoned:
#             status = TaskStatus.ABANDONED
#         else:
#             status = TaskStatus.FAILED

#         # Build metrics dict
#         metrics = {
#             'status': status.value,
#             'required_slots_filled': len(filled_required),
#             'required_slots_total': len(required),
#             'required_completion_rate': required_completion,
#             'optional_slots_filled': len(filled_optional),
#             'optional_slots_total': len(optional),
#             'optional_completion_rate': optional_completion,
#             'task_successful': task_successful,
#             'user_abandoned': user_abandoned,
#             'extracted_slots': extracted,
#             'slot_breakdown': {
#                 slot.name: {
#                     'value': slot.value,
#                     'filled': slot.status == SlotStatus.FILLED,
#                     'required': slot.required
#                 }
#                 for slot in slots.get_all_slots()
#             }
#         }

#         return slots, metrics

#     def evaluate_session(self, session: TaskSession) -> TaskSession:
#         """
#         Evaluate a single session and fill slots from dialogue

#         Args:
#             session: TaskSession with dialogue turns

#         Returns:
#             Updated TaskSession with filled slots
#         """
#         slots = session.slots

#         for turn_idx, turn in enumerate(session.turns):
#             if turn.speaker == 'user':
#                 # Extract slots from user utterance
#                 extracted = self.extract_slots_from_text(turn.text)
#                 turn.extracted_slots = extracted

#                 # Update slot values
#                 for slot_name, value in extracted.items():
#                     slot = getattr(slots, slot_name, None)
#                     if slot:
#                         if slot.status == SlotStatus.MISSING:
#                             slot.value = value
#                             slot.status = SlotStatus.FILLED
#                             slot.turn_filled = turn_idx
#                         elif slot.value != value:
#                             slot.value = value
#                             slot.changed = True

#         # Determine session status
#         required = slots.get_required_slots()
#         filled_required = [s for s in required if s.status == SlotStatus.FILLED]

#         if len(filled_required) == len(required):
#             session.status = TaskStatus.COMPLETED
#         elif len(filled_required) > 0:
#             session.status = TaskStatus.PARTIAL
#         elif session.abandoned:
#             session.status = TaskStatus.ABANDONED
#         else:
#             session.status = TaskStatus.FAILED

#         return session

#     def calculate_kappa_statistic(self,
#                                   observed_agreement: float,
#                                   expected_agreement: float) -> float:
#         """
#         Calculate Cohen's Kappa for information exchange

#         κ = (P(A) - P(E)) / (1 - P(E))

#         Args:
#             observed_agreement: Actual agreement rate
#             expected_agreement: Expected agreement by chance

#         Returns:
#             Kappa statistic
#         """
#         if expected_agreement >= 1.0:
#             return 0.0

#         kappa = (observed_agreement - expected_agreement) / (1 - expected_agreement)
#         return max(0.0, min(1.0, kappa))

#     def estimate_user_satisfaction(self,
#                                    task_success: float,
#                                    dialogue_cost: float,
#                                    kappa: float) -> float:
#         """
#         Estimate user satisfaction using PARADISE framework

#         Satisfaction = α·Success - β·Cost + γ·Kappa

#         Args:
#             task_success: Task completion score (0-1)
#             dialogue_cost: Normalized dialogue length
#             kappa: Information exchange quality

#         Returns:
#             Estimated satisfaction (0-1)
#         """
#         # PARADISE coefficients (from Walker et al., 1997)
#         alpha = 0.5  # Weight for task success
#         beta = 0.2   # Weight for efficiency (cost)
#         gamma = 0.3  # Weight for information quality

#         satisfaction = (alpha * task_success -
#                        beta * dialogue_cost +
#                        gamma * kappa)

#         return max(0.0, min(1.0, satisfaction))

#     def detect_false_containment(self, session: TaskSession) -> bool:
#         """
#         Detect false containment (user abandons without completion)

#         Indicators:
#         - Few turns (< 3)
#         - Low slot filling rate (< 50%)
#         - No explicit completion confirmation
#         - Abrupt ending

#         Args:
#             session: Task session

#         Returns:
#             True if likely false containment
#         """
#         if session.status == TaskStatus.COMPLETED:
#             return False

#         # Check indicators
#         num_turns = len(session.turns)
#         required_slots = session.slots.get_required_slots()
#         filled_slots = [s for s in required_slots if s.status == SlotStatus.FILLED]
#         filling_rate = len(filled_slots) / len(required_slots) if required_slots else 0

#         # Last turn analysis
#         last_turn = session.turns[-1] if session.turns else None
#         abrupt_ending = False
#         if last_turn:
#             # Check for frustration or abrupt endings
#             frustration_keywords = ['nevermind', 'forget it', 'not working', 'bye', 'goodbye']
#             abrupt_ending = any(kw in last_turn.text.lower() for kw in frustration_keywords)

#         # False containment if:
#         # 1. Short dialogue (< 3 turns)
#         # 2. Low filling rate (< 50%)
#         # 3. Abrupt ending OR marked as abandoned
#         is_false_containment = (
#             num_turns < 3 or
#             (filling_rate < 0.5 and (abrupt_ending or session.abandoned))
#         )

#         return is_false_containment

#     def evaluate_all_sessions(self,
#                              sessions: List[TaskSession],
#                              target_tsr: float = 0.85,
#                              target_containment: float = 0.80
#                              ) -> TaskCompletionMetrics:
#         """
#         Evaluate multiple sessions and calculate overall metrics

#         Args:
#             sessions: List of task sessions
#             target_tsr: Target TSR for comparison (default 85%)
#             target_containment: Target containment (default 80%)

#         Returns:
#             TaskCompletionMetrics with all results
#         """
#         print(f"\n{'='*70}")
#         print("TASK COMPLETION EVALUATION")
#         print(f"{'='*70}")
#         print(f"Analyzing {len(sessions)} sessions...")

#         # Process each session
#         for session in sessions:
#             self.evaluate_session(session)

#         # Count task outcomes
#         completed = len([s for s in sessions if s.status == TaskStatus.COMPLETED])
#         partial = len([s for s in sessions if s.status == TaskStatus.PARTIAL])
#         failed = len([s for s in sessions if s.status == TaskStatus.FAILED])
#         abandoned = len([s for s in sessions if s.status == TaskStatus.ABANDONED])

#         # Calculate TSR
#         tsr = (completed / len(sessions)) * 100 if sessions else 0.0

#         # Slot filling stats
#         all_required_filled = 0
#         all_required_total = 0
#         all_optional_filled = 0
#         all_optional_total = 0

#         for session in sessions:
#             required = session.slots.get_required_slots()
#             optional = session.slots.get_optional_slots()

#             all_required_total += len(required)
#             all_required_filled += len([s for s in required if s.status == SlotStatus.FILLED])

#             all_optional_total += len(optional)
#             all_optional_filled += len([s for s in optional if s.status == SlotStatus.FILLED])

#         slot_filling_rate = (all_required_filled / all_required_total * 100) if all_required_total > 0 else 0.0

#         # Containment metrics
#         escalated = len([s for s in sessions if s.escalated])
#         containment_rate = ((len(sessions) - escalated) / len(sessions)) * 100 if sessions else 0.0

#         # First-call resolution
#         first_call = len([s for s in sessions if s.status == TaskStatus.COMPLETED and not s.escalated])
#         fcr = (first_call / len(sessions)) * 100 if sessions else 0.0

#         # False containment
#         false_containment_count = len([s for s in sessions if self.detect_false_containment(s)])
#         false_containment_rate = (false_containment_count / len(sessions)) * 100 if sessions else 0.0

#         # Efficiency metrics
#         completed_sessions = [s for s in sessions if s.status == TaskStatus.COMPLETED]
#         avg_turns = np.mean([len(s.turns) for s in completed_sessions]) if completed_sessions else 0

#         total_slots_filled = all_required_filled + all_optional_filled
#         total_turns = sum(len(s.turns) for s in sessions)
#         avg_slots_per_turn = total_slots_filled / total_turns if total_turns > 0 else 0

#         # Slot correction rate
#         total_changed = sum(len([s for s in session.slots.get_all_slots() if s.changed])
#                           for session in sessions)
#         slot_correction_rate = (total_changed / total_slots_filled) * 100 if total_slots_filled > 0 else 0

#         # PARADISE framework metrics
#         # Calculate average metrics across sessions
#         kappas = []
#         dialogue_costs = []
#         satisfactions = []

#         for session in sessions:
#             # Observed agreement = slot filling rate
#             required = session.slots.get_required_slots()
#             filled = [s for s in required if s.status == SlotStatus.FILLED]
#             observed = len(filled) / len(required) if required else 0

#             # Expected agreement (baseline = random guessing)
#             expected = 1.0 / len(required) if required else 0.5

#             kappa = self.calculate_kappa_statistic(observed, expected)
#             kappas.append(kappa)

#             # Dialogue cost (normalized by turns)
#             cost = min(len(session.turns) / 20.0, 1.0)  # Normalize to max 20 turns
#             dialogue_costs.append(cost)

#             # Task success
#             success = 1.0 if session.status == TaskStatus.COMPLETED else 0.0

#             # Satisfaction
#             satisfaction = self.estimate_user_satisfaction(success, cost, kappa)
#             satisfactions.append(satisfaction)

#         avg_kappa = np.mean(kappas) if kappas else 0.0
#         avg_dialogue_cost = np.mean(dialogue_costs) if dialogue_costs else 0.0
#         avg_satisfaction = np.mean(satisfactions) if satisfactions else 0.0

#         # Detailed slot breakdown
#         slot_details = {}
#         for slot_name in self.required_slots + self.optional_slots:
#             filled_count = 0
#             total_count = 0

#             for session in sessions:
#                 slot = getattr(session.slots, slot_name, None)
#                 if slot:
#                     total_count += 1
#                     if slot.status == SlotStatus.FILLED:
#                         filled_count += 1

#             slot_details[slot_name] = {
#                 'filled': filled_count,
#                 'total': total_count,
#                 'fill_rate': (filled_count / total_count * 100) if total_count > 0 else 0
#             }

#         # Task breakdown
#         task_breakdown = {
#             'completed': completed,
#             'partial': partial,
#             'failed': failed,
#             'abandoned': abandoned,
#             'escalated': escalated,
#             'false_containment': false_containment_count
#         }

#         metrics = TaskCompletionMetrics(
#             tsr=tsr,
#             total_tasks=len(sessions),
#             completed_tasks=completed,
#             partial_tasks=partial,
#             failed_tasks=failed,
#             abandoned_tasks=abandoned,
#             required_slots_filled=all_required_filled,
#             required_slots_total=all_required_total,
#             optional_slots_filled=all_optional_filled,
#             optional_slots_total=all_optional_total,
#             slot_filling_rate=slot_filling_rate,
#             kappa_statistic=avg_kappa,
#             dialogue_success=completed / len(sessions) if sessions else 0,
#             dialogue_cost=avg_dialogue_cost,
#             user_satisfaction_estimate=avg_satisfaction,
#             first_call_resolution=fcr,
#             containment_rate=containment_rate,
#             false_containment_rate=false_containment_rate,
#             avg_turns_to_completion=avg_turns,
#             avg_slots_per_turn=avg_slots_per_turn,
#             slot_correction_rate=slot_correction_rate,
#             slot_details=slot_details,
#             task_breakdown=task_breakdown
#         )

#         # Print summary
#         print(f"\n✓ Evaluation Complete:")
#         print(f"  TSR: {tsr:.2f}% (target: {target_tsr*100:.0f}%)")
#         print(f"  First-Call Resolution: {fcr:.2f}% (target: 85%)")
#         print(f"  Containment: {containment_rate:.2f}% (target: {target_containment*100:.0f}%)")
#         print(f"  False Containment: {false_containment_rate:.2f}%")
#         print(f"  User Satisfaction (est): {avg_satisfaction:.2f}")

#         return metrics


# def format_single_session_report(slots: TravelAgentSlots, metrics: Dict[str, Any]) -> str:
#     """Format a single session's task completion into readable report"""

#     status = metrics.get('status', 'unknown')
#     status_emoji = {
#         'completed': '✅',
#         'partial': '◐',
#         'failed': '❌',
#         'abandoned': '⊗'
#     }.get(status, '?')

#     required_pct = metrics.get('required_completion_rate', 0.0) * 100
#     optional_pct = metrics.get('optional_completion_rate', 0.0) * 100

#     report = f"""
# ╔══════════════════════════════════════════════════════════════════╗
# ║              TASK COMPLETION - SINGLE SESSION                     ║
# ╚══════════════════════════════════════════════════════════════════╝

# 📊 SESSION STATUS
# ──────────────────────────────────────────────────────────────────
#   {status_emoji} Status: {status.upper()}
#   Task Successful: {'Yes' if metrics.get('task_successful') else 'No'}
#   User Abandoned: {'Yes' if metrics.get('user_abandoned') else 'No'}

# 🎯 SLOT FILLING
# ──────────────────────────────────────────────────────────────────
#   Required Slots: {metrics.get('required_slots_filled', 0)}/{metrics.get('required_slots_total', 0)} ({required_pct:.1f}%)
#   Optional Slots: {metrics.get('optional_slots_filled', 0)}/{metrics.get('optional_slots_total', 0)} ({optional_pct:.1f}%)

# 📋 EXTRACTED INFORMATION
# ──────────────────────────────────────────────────────────────────"""

#     for slot in slots.get_all_slots():
#         fill_status = "✓" if slot.status == SlotStatus.FILLED else "✗"
#         req_label = "(required)" if slot.required else "(optional)"
#         value_str = slot.value if slot.value else "—"
#         report += f"\n  {fill_status} {slot.name:20s} {req_label:12s}: {value_str}"

#     if required_pct < 100:
#         report += "\n\n⚠️  WARNING: Not all required slots filled - task cannot be completed"

#     return report


# def format_task_completion_report(metrics: TaskCompletionMetrics) -> str:
#     """Format task completion metrics into readable report"""

#     tsr_status = "✅" if metrics.tsr >= 85 else "⚠️" if metrics.tsr >= 70 else "❌"
#     fcr_status = "✅" if metrics.first_call_resolution >= 85 else "⚠️" if metrics.first_call_resolution >= 70 else "❌"
#     containment_status = "✅" if metrics.containment_rate >= 80 else "⚠️" if metrics.containment_rate >= 65 else "❌"

#     report = f"""
# ╔══════════════════════════════════════════════════════════════════╗
# ║           TASK COMPLETION & SUCCESS RATE REPORT                   ║
# ╚══════════════════════════════════════════════════════════════════╝

# 📊 TASK SUCCESS RATE (TSR)
# ──────────────────────────────────────────────────────────────────
#   {tsr_status} TSR:                      {metrics.tsr:.2f}%
#   Total Tasks:              {metrics.total_tasks}
#   ✓ Completed:              {metrics.completed_tasks} ({metrics.completed_tasks/metrics.total_tasks*100 if metrics.total_tasks else 0:.1f}%)
#   ◐ Partial:                {metrics.partial_tasks} ({metrics.partial_tasks/metrics.total_tasks*100 if metrics.total_tasks else 0:.1f}%)
#   ✗ Failed:                 {metrics.failed_tasks} ({metrics.failed_tasks/metrics.total_tasks*100 if metrics.total_tasks else 0:.1f}%)
#   ⊗ Abandoned:              {metrics.abandoned_tasks} ({metrics.abandoned_tasks/metrics.total_tasks*100 if metrics.total_tasks else 0:.1f}%)

# 🎯 SLOT FILLING METRICS
# ──────────────────────────────────────────────────────────────────
#   Required Slots Filled:    {metrics.required_slots_filled}/{metrics.required_slots_total} ({metrics.slot_filling_rate:.2f}%)
#   Optional Slots Filled:    {metrics.optional_slots_filled}/{metrics.optional_slots_total}
#   Avg Slots per Turn:       {metrics.avg_slots_per_turn:.2f}
#   Slot Correction Rate:     {metrics.slot_correction_rate:.2f}%

# 📋 SLOT BREAKDOWN
# ──────────────────────────────────────────────────────────────────"""

#     for slot_name, details in sorted(metrics.slot_details.items()):
#         fill_status = "✓" if details['fill_rate'] >= 90 else "◐" if details['fill_rate'] >= 70 else "✗"
#         report += f"\n  {fill_status} {slot_name:20s}: {details['filled']:>3}/{details['total']:<3} ({details['fill_rate']:>5.1f}%)"

#     report += f"""

# 🎭 PARADISE FRAMEWORK
# ──────────────────────────────────────────────────────────────────
#   Kappa Statistic (κ):      {metrics.kappa_statistic:.4f}
#   Dialogue Success:         {metrics.dialogue_success:.4f}
#   Dialogue Cost:            {metrics.dialogue_cost:.4f}
#   User Satisfaction (est):  {metrics.user_satisfaction_estimate:.4f}

# 📞 CONTAINMENT METRICS
# ──────────────────────────────────────────────────────────────────
#   {fcr_status} First-Call Resolution:  {metrics.first_call_resolution:.2f}% (target: 85%+)
#   {containment_status} Containment Rate:       {metrics.containment_rate:.2f}% (target: 80%+)
#   ⚠️  False Containment:      {metrics.false_containment_rate:.2f}%

# ⏱️  EFFICIENCY METRICS
# ──────────────────────────────────────────────────────────────────
#   Avg Turns to Completion:  {metrics.avg_turns_to_completion:.2f}
#   Avg Slots per Turn:       {metrics.avg_slots_per_turn:.2f}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 📖 METRICS GUIDE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# TSR (Task Success Rate)
#   Formula: (Completed Tasks / Total Tasks) × 100
#   Target: 85%+ for well-defined transactional flows

# PARADISE Framework
#   Kappa (κ): Information exchange quality
#     Formula: (P(A) - P(E)) / (1 - P(E))
#     Range: 0-1, higher is better

#   User Satisfaction: α·Success - β·Cost + γ·Kappa
#     Estimated using task success, dialogue efficiency, and info quality

# First-Call Resolution (FCR)
#   Tasks completed on first attempt without escalation
#   Industry target: 85%+

# Containment Rate
#   Tasks handled without human escalation
#   Industry target: 80%+

# False Containment
#   Users abandoning without completion
#   High containment + low CSAT = false containment
#   Warning sign: Users giving up rather than being helped

# Slot Filling Rate
#   Percentage of required information captured
#   100% required for task completion
# """

#     # Warnings section
#     warnings_text = []

#     if metrics.tsr < 70:
#         warnings_text.append("⚠️  TSR below 70% - critical issue")
#     if metrics.false_containment_rate > 20:
#         warnings_text.append("⚠️  High false containment - users abandoning")
#     if metrics.containment_rate < 65:
#         warnings_text.append("⚠️  Low containment - high escalation rate")
#     if metrics.user_satisfaction_estimate < 0.6:
#         warnings_text.append("⚠️  Low estimated user satisfaction")

#     if warnings_text:
#         report += "\n⚠️  WARNINGS\n"
#         report += "──────────────────────────────────────────────────────────────────\n"
#         for warning in warnings_text:
#             report += f"  {warning}\n"

#     return report


# def main():
#     """Example usage - supports both single transcript and batch evaluation"""
#     import sys

#     if len(sys.argv) < 2:
#         print("Usage:")
#         print("  Single transcript:  python task_completion.py <transcript.txt>")
#         print("  Batch sessions:     python task_completion.py --batch <sessions.json>")
#         sys.exit(1)

#     evaluator = TaskCompletionEvaluator()

#     # Batch mode
#     if sys.argv[1] == '--batch':
#         if len(sys.argv) < 3:
#             print("Error: --batch requires sessions JSON file")
#             sys.exit(1)

#         # Load sessions from JSON
#         with open(sys.argv[2], 'r') as f:
#             sessions_data = json.load(f)

#         # Convert to TaskSession objects
#         sessions = []
#         for sess_data in sessions_data:
#             # Create session (simplified - adapt to your data format)
#             session = TaskSession(
#                 session_id=sess_data['id'],
#                 turns=[],
#                 slots=TravelAgentSlots(),
#                 status=TaskStatus.PARTIAL
#             )
#             # Add turns, etc.
#             sessions.append(session)

#         # Evaluate
#         metrics = evaluator.evaluate_all_sessions(sessions)

#         # Print report
#         print(format_task_completion_report(metrics))

#         # Save results
#         with open('task_completion_results.json', 'w') as f:
#             json.dump(asdict(metrics), f, indent=2)

#     # Single transcript mode
#     else:
#         transcript_path = sys.argv[1]

#         # Read transcript
#         with open(transcript_path, 'r') as f:
#             transcript = f.read()

#         print(f"\nEvaluating transcript: {transcript_path}")
#         print("="*70)

#         # Evaluate
#         slots, metrics = evaluator.evaluate_transcript(transcript)

#         # Print report
#         print(format_single_session_report(slots, metrics))

#         # Save results
#         output_path = transcript_path.replace('.txt', '_task_completion.json')
#         with open(output_path, 'w') as f:
#             json.dump(metrics, f, indent=2, default=str)

#         print(f"\n💾 Results saved to: {output_path}")


# if __name__ == "__main__":
#     main()


"""
Task Success Rate (TSR) & Completion Metrics - UPDATED VERSION
Implements PARADISE framework, slot-filling tracking, and containment metrics
Designed for goal-oriented dialogues like travel booking agents

IMPROVEMENTS:
- Better regex patterns with context awareness
- Negation handling ("not a hotel")
- Multiple passes for better accuracy
- Entity disambiguation
- Confidence scoring
"""

import json
import re
import numpy as np
from typing import List, Dict, Set, Optional, Tuple, Any
from dataclasses import dataclass, asdict, field
from enum import Enum
import warnings
warnings.filterwarnings('ignore')


class TaskStatus(Enum):
    """Task completion status"""
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"
    ABANDONED = "abandoned"


class SlotStatus(Enum):
    """Individual slot status"""
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    MISSING = "missing"
    INCORRECT = "incorrect"


@dataclass
class Slot:
    """Represents a required information slot"""
    name: str
    required: bool
    value: Optional[str] = None
    status: SlotStatus = SlotStatus.MISSING
    confidence: float = 0.0
    turn_filled: int = -1  # Which turn it was filled
    changed: bool = False  # Was it changed after initial fill
    extraction_method: str = ""  # How it was extracted

    @property
    def filled(self) -> bool:
        """Convenience property to check if slot is filled"""
        return self.status == SlotStatus.FILLED

    @filled.setter
    def filled(self, value: bool):
        """Set slot status based on filled boolean"""
        if value:
            self.status = SlotStatus.FILLED
        else:
            self.status = SlotStatus.MISSING


@dataclass
class TravelAgentSlots:
    """Specific slots for travel agent task"""
    # Required slots
    destination: Slot = field(default_factory=lambda: Slot("destination", required=True))
    checkin_date: Slot = field(default_factory=lambda: Slot("checkin_date", required=True))
    checkout_date: Slot = field(default_factory=lambda: Slot("checkout_date", required=True))
    num_guests: Slot = field(default_factory=lambda: Slot("num_guests", required=True))
    budget_min: Slot = field(default_factory=lambda: Slot("budget_min", required=True))
    budget_max: Slot = field(default_factory=lambda: Slot("budget_max", required=True))

    # Optional slots
    accommodation_type: Slot = field(default_factory=lambda: Slot("accommodation_type", required=False))
    amenities: Slot = field(default_factory=lambda: Slot("amenities", required=False))
    neighborhood: Slot = field(default_factory=lambda: Slot("neighborhood", required=False))
    special_requests: Slot = field(default_factory=lambda: Slot("special_requests", required=False))

    def get_all_slots(self) -> List[Slot]:
        """Get all slot objects"""
        return [
            self.destination, self.checkin_date, self.checkout_date,
            self.num_guests, self.budget_min, self.budget_max,
            self.accommodation_type, self.amenities,
            self.neighborhood, self.special_requests
        ]

    def get_required_slots(self) -> List[Slot]:
        """Get only required slots"""
        return [s for s in self.get_all_slots() if s.required]

    def get_optional_slots(self) -> List[Slot]:
        """Get only optional slots"""
        return [s for s in self.get_all_slots() if not s.required]


@dataclass
class TaskCompletionMetrics:
    """Complete task success metrics"""
    # Task Success Rate (TSR)
    tsr: float  # Overall TSR percentage
    total_tasks: int
    completed_tasks: int
    partial_tasks: int
    failed_tasks: int
    abandoned_tasks: int

    # Slot filling metrics
    required_slots_filled: int
    required_slots_total: int
    optional_slots_filled: int
    optional_slots_total: int
    slot_filling_rate: float  # Percentage of required slots filled

    # PARADISE framework metrics
    kappa_statistic: float  # Information exchange quality
    dialogue_success: float  # Binary or scaled success
    dialogue_cost: float  # Number of turns / efficiency
    user_satisfaction_estimate: float  # Estimated from PARADISE model

    # Containment metrics
    first_call_resolution: float  # Percentage resolved in first attempt
    containment_rate: float  # Percentage handled without escalation
    false_containment_rate: float  # Abandoned without completion

    # Efficiency metrics
    avg_turns_to_completion: float
    avg_slots_per_turn: float
    slot_correction_rate: float  # How often slots needed correction

    # Detailed breakdown
    slot_details: Dict[str, Dict[str, Any]]
    task_breakdown: Dict[str, int]


@dataclass
class DialogueTurn:
    """Represents a single dialogue turn"""
    turn_id: int
    speaker: str  # 'user' or 'agent'
    text: str
    extracted_slots: Dict[str, str]
    timestamp: float = 0.0


@dataclass
class TaskSession:
    """Complete task session"""
    session_id: str
    turns: List[DialogueTurn]
    slots: TravelAgentSlots
    status: TaskStatus
    escalated: bool = False
    abandoned: bool = False
    user_satisfied: Optional[bool] = None
    completion_time: float = 0.0


class TaskCompletionEvaluator:
    """
    Evaluates task completion for goal-oriented dialogues

    Tracks:
    - Task Success Rate (TSR)
    - Slot filling completion
    - PARADISE framework metrics
    - Containment and escalation
    - False containment detection
    """

    def __init__(self,
                 required_slots: Optional[List[str]] = None,
                 optional_slots: Optional[List[str]] = None):
        """
        Initialize evaluator

        Args:
            required_slots: List of required slot names
            optional_slots: List of optional slot names
        """
        # Default travel agent slots
        self.default_required = [
            'destination', 'checkin_date', 'checkout_date',
            'num_guests', 'budget_min', 'budget_max'
        ]

        self.default_optional = [
            'accommodation_type', 'amenities',
            'neighborhood', 'special_requests'
        ]

        self.required_slots = required_slots or self.default_required
        self.optional_slots = optional_slots or self.default_optional

        # Extraction patterns for travel agent
        self._init_extraction_patterns()
        
        # Common city abbreviations
        self.city_expansions = {
            'SF': 'San Francisco',
            'NYC': 'New York City',
            'LA': 'Los Angeles',
            'DC': 'Washington DC',
        }

    def _init_extraction_patterns(self):
        """Initialize regex patterns for slot extraction with improved accuracy"""
        self.patterns = {
            'destination': [
                # Priority: Explicit mentions
                r'(?:going|travel(?:ing)?|visit(?:ing)?|staying)\s+(?:to|in)\s+([A-Z][a-zA-Z\s]+?)(?:\s+in|\s+from|\s+for|\.|,|$)',
                # City names (capitalized)
                r'\b(San Francisco|New York|Los Angeles|Chicago|Boston|Seattle|Portland|Austin|Miami|Denver)\b',
                # Abbreviations
                r'\b(SF|NYC|LA|DC)\b(?!\s*in\s+(January|February|March|April|May|June|July|August|September|October|November|December))',
                # With prepositions
                r'(?:^|\s)(?:to|in|at)\s+([A-Z][a-zA-Z\s]{2,})(?:\s+in\s+\w+|\s+from|\.|,)',
            ],
            'checkin_date': [
                # "from Feb 15th" or "arriving Feb 15th"
                r'(?:from|check-?in|arrive|arriving|start(?:ing)?)\s+(?:on\s+)?([A-Z][a-z]+\s+\d{1,2}(?:st|nd|rd|th)?)',
                # "Feb 15th to Feb 20th"
                r'([A-Z][a-z]+\s+\d{1,2}(?:st|nd|rd|th)?)\s+(?:to|until|-)',
                # Date formats
                r'(\d{1,2}/\d{1,2}(?:/\d{2,4})?)\s+(?:to|until)',
            ],
            'checkout_date': [
                # "to Feb 20th" or "leaving Feb 20th"
                r'(?:to|until|check-?out|leave|leaving|end(?:ing)?)\s+(?:on\s+)?([A-Z][a-z]+\s+\d{1,2}(?:st|nd|rd|th)?)',
                # After "to" in range
                r'(?:to|until|-)\s+([A-Z][a-z]+\s+\d{1,2}(?:st|nd|rd|th)?)',
                # Date formats
                r'(?:to|until)\s+(\d{1,2}/\d{1,2}(?:/\d{2,4})?)',
            ],
            'num_guests': [
                # "three of us", "3 people"
                r'(?:^|\s)(one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+(?:of\s+us|people|guests|adults|travelers)',
                # "party of 3"
                r'(?:for|party of)\s+(one|two|three|four|five|six|seven|eight|nine|ten|\d+)',
                # Just number with context
                r'(?:^|\s)(one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+(?:of\s+us|people)',
            ],
            'budget_min': [
                # "around $150"
                r'around\s+\$?(\d+)',
                # "$150 to $200"
                r'\$?(\d+)\s*(?:to|-)\s*\$?\d+',
                # "budget of $150"
                r'budget.*?\$?(\d+)',
            ],
            'budget_max': [
                # "stretch up to $200", "up to $200"
                r'(?:stretch|up).*?(?:to|up to)\s+\$?(\d+)',
                # "$150 to $200"
                r'\$?\d+\s*(?:to|-)\s*\$?(\d+)',
                # "max $200"
                r'max(?:imum)?\s+\$?(\d+)',
            ],
            'accommodation_type': [
                # Positive mentions (looking FOR)
                r'(?:looking for|prefer|want|interested in).*?(hotel|airbnb|hostel|resort|apartment|villa|b&b|motel)',
                # Direct mentions (avoiding negations)
                r'(?:^|[^not]\s)(hotel|airbnb|hostel|resort|apartment|villa|b&b|motel)(?:\s|$)',
                # "more of an X"
                r'more of (?:a|an)\s+(hotel|airbnb|hostel|resort|apartment|villa|b&b|motel)',
            ],
            'amenities': [
                r'(?:with|need|want|require|includes?).*?(pool|gym|wifi|parking|kitchen|breakfast|spa|beach|laundry)',
                r'(pool|gym|wifi|parking|kitchen|breakfast|spa|beach|laundry)',
            ],
            'neighborhood': [
                # Known areas
                r'(?:south of market|soma|downtown|financial district|mission|castro|haight|nob hill|pacific heights)',
                # "in the X area/district"
                r'(?:in|near|around)\s+(?:the\s+)?([A-Z][a-zA-Z\s]+?)(?:\s+area|\s+district)',
            ],
        }

    def check_negation(self, text: str, match_start: int, match_end: int) -> bool:
        """
        Check if matched text is negated
        
        Args:
            text: Full text
            match_start: Start position of match
            match_end: End position of match
            
        Returns:
            True if negated, False otherwise
        """
        # Look back up to 50 characters for negation words
        lookback_start = max(0, match_start - 50)
        context = text[lookback_start:match_end].lower()
        
        negation_words = ['not', "not a", 'no', "don't", 'without', 'except', 'excluding']
        
        for neg_word in negation_words:
            if neg_word in context:
                # Check if negation is close enough (within 10 words)
                neg_pos = context.rfind(neg_word)
                words_between = len(context[neg_pos:].split())
                if words_between <= 10:
                    return True
        
        return False

    def extract_with_confidence(self, text: str, patterns: List[str], 
                                slot_name: str) -> Tuple[Optional[str], float, str]:
        """
        Extract slot value with confidence score
        
        Args:
            text: Input text
            patterns: List of regex patterns
            slot_name: Name of slot being extracted
            
        Returns:
            Tuple of (value, confidence, method)
        """
        matches = []
        
        for pattern_idx, pattern in enumerate(patterns):
            for match in re.finditer(pattern, text, re.IGNORECASE):
                value = match.group(1).strip() if match.lastindex else match.group(0).strip()
                
                # Check for negation
                if slot_name == 'accommodation_type':
                    if self.check_negation(text, match.start(), match.end()):
                        continue  # Skip negated matches
                
                # Calculate confidence based on pattern priority and context
                base_confidence = 1.0 - (pattern_idx * 0.1)  # Earlier patterns = higher confidence
                
                # Boost confidence for certain indicators
                context_window = text[max(0, match.start()-100):min(len(text), match.end()+100)]
                
                if slot_name == 'destination':
                    # Higher confidence if capitalized and not in common phrase
                    if value[0].isupper() and value not in ['February', 'March', 'April', 'May', 'June', 'July']:
                        base_confidence += 0.2
                    # Lower confidence if it's in a question
                    if 'you need' in context_window.lower() or 'can I help' in context_window.lower():
                        base_confidence -= 0.5
                
                elif slot_name == 'accommodation_type':
                    # Higher confidence if preceded by "looking for", "want", "prefer"
                    if any(word in context_window.lower() for word in ['looking for', 'want', 'prefer', 'more of']):
                        base_confidence += 0.3
                
                confidence = max(0.0, min(1.0, base_confidence))
                
                matches.append({
                    'value': value,
                    'confidence': confidence,
                    'pattern_idx': pattern_idx,
                    'position': match.start()
                })
        
        if not matches:
            return None, 0.0, "not_found"
        
        # Sort by confidence, then by position (later in text often better)
        matches.sort(key=lambda x: (-x['confidence'], -x['position']))
        
        best_match = matches[0]
        method = f"pattern_{best_match['pattern_idx']}"
        
        return best_match['value'], best_match['confidence'], method

    def normalize_value(self, slot_name: str, value: str) -> str:
        """Normalize extracted values"""
        
        if slot_name == 'destination':
            # Expand abbreviations
            value_upper = value.upper()
            if value_upper in self.city_expansions:
                return self.city_expansions[value_upper]
        
        elif slot_name == 'num_guests':
            # Convert words to numbers
            word_to_num = {
                'one': '1', 'two': '2', 'three': '3', 'four': '4',
                'five': '5', 'six': '6', 'seven': '7', 'eight': '8',
                'nine': '9', 'ten': '10'
            }
            value_lower = value.lower()
            if value_lower in word_to_num:
                return word_to_num[value_lower]
        
        elif slot_name == 'accommodation_type':
            # Standardize accommodation types
            value_lower = value.lower()
            type_map = {
                'b&b': 'Bed & Breakfast',
                'airbnb': 'Airbnb',
                'hotel': 'Hotel',
                'hostel': 'Hostel',
                'resort': 'Resort',
                'apartment': 'Apartment',
                'villa': 'Villa',
            }
            return type_map.get(value_lower, value.title())
        
        elif slot_name == 'neighborhood':
            # Standardize neighborhood names
            value_lower = value.lower()
            if 'soma' in value_lower or 'south of market' in value_lower:
                return 'SoMa (South of Market)'
        
        return value

    def extract_slots_from_text(self, text: str, verbose: bool = False) -> Dict[str, Any]:
        """
        Extract slot values from text using improved patterns
        
        Args:
            text: Input text
            verbose: Print extraction details
            
        Returns:
            Dictionary with extracted slot values and metadata
        """
        extracted = {}
        extraction_details = {}
        
        for slot_name, patterns in self.patterns.items():
            value, confidence, method = self.extract_with_confidence(text, patterns, slot_name)
            
            if value:
                # Normalize the value
                normalized_value = self.normalize_value(slot_name, value)
                
                extracted[slot_name] = normalized_value
                extraction_details[slot_name] = {
                    'value': normalized_value,
                    'raw_value': value,
                    'confidence': confidence,
                    'method': method
                }
                
                if verbose:
                    print(f"✓ {slot_name:20s}: '{normalized_value}' (confidence: {confidence:.2f}, method: {method})")
        
        return {
            'slots': extracted,
            'details': extraction_details
        }

    def evaluate_transcript(self,
                           transcript: str,
                           task_successful: bool = True,
                           user_abandoned: bool = False,
                           verbose: bool = True) -> Tuple[TravelAgentSlots, Dict[str, Any]]:
        """
        Simple evaluation from just a transcript string
        
        Args:
            transcript: Full dialogue transcript
            task_successful: Whether task was completed
            user_abandoned: Whether user abandoned
            verbose: Print extraction details
            
        Returns:
            Tuple of (filled slots, metrics dict)
        """
        if verbose:
            print("\n" + "="*70)
            print("EXTRACTING SLOTS FROM TRANSCRIPT")
            print("="*70 + "\n")
        
        # Extract all slots from transcript
        extraction_result = self.extract_slots_from_text(transcript, verbose=verbose)
        extracted = extraction_result['slots']
        details = extraction_result['details']

        # Create slots object
        slots = TravelAgentSlots()

        # Fill slots
        for slot_name, value in extracted.items():
            slot = getattr(slots, slot_name, None)
            if slot:
                slot.value = value
                slot.status = SlotStatus.FILLED
                slot.confidence = details[slot_name]['confidence']
                slot.extraction_method = details[slot_name]['method']

        # Calculate completion metrics
        required = slots.get_required_slots()
        filled_required = [s for s in required if s.status == SlotStatus.FILLED]
        required_completion = len(filled_required) / len(required) if required else 0.0

        optional = slots.get_optional_slots()
        filled_optional = [s for s in optional if s.status == SlotStatus.FILLED]
        optional_completion = len(filled_optional) / len(optional) if optional else 0.0

        # Determine status
        if len(filled_required) == len(required) and task_successful:
            status = TaskStatus.COMPLETED
        elif len(filled_required) > 0:
            status = TaskStatus.PARTIAL
        elif user_abandoned:
            status = TaskStatus.ABANDONED
        else:
            status = TaskStatus.FAILED

        # Build metrics dict
        metrics = {
            'status': status.value,
            'required_slots_filled': len(filled_required),
            'required_slots_total': len(required),
            'required_completion_rate': required_completion,
            'optional_slots_filled': len(filled_optional),
            'optional_slots_total': len(optional),
            'optional_completion_rate': optional_completion,
            'task_successful': task_successful,
            'user_abandoned': user_abandoned,
            'extracted_slots': extracted,
            'extraction_details': details,
            'slot_breakdown': {
                slot.name: {
                    'value': slot.value,
                    'filled': slot.status == SlotStatus.FILLED,
                    'required': slot.required,
                    'confidence': slot.confidence,
                    'method': slot.extraction_method
                }
                for slot in slots.get_all_slots()
            }
        }

        return slots, metrics

    def evaluate_session(self, session: TaskSession) -> TaskSession:
        """
        Evaluate a single session and fill slots from dialogue
        
        Args:
            session: TaskSession with dialogue turns
            
        Returns:
            Updated TaskSession with filled slots
        """
        slots = session.slots

        for turn_idx, turn in enumerate(session.turns):
            if turn.speaker == 'user':
                # Extract slots from user utterance
                result = self.extract_slots_from_text(turn.text)
                extracted = result['slots']
                turn.extracted_slots = extracted

                # Update slot values
                for slot_name, value in extracted.items():
                    slot = getattr(slots, slot_name, None)
                    if slot:
                        if slot.status == SlotStatus.MISSING:
                            slot.value = value
                            slot.status = SlotStatus.FILLED
                            slot.turn_filled = turn_idx
                        elif slot.value != value:
                            slot.value = value
                            slot.changed = True

        # Determine session status
        required = slots.get_required_slots()
        filled_required = [s for s in required if s.status == SlotStatus.FILLED]

        if len(filled_required) == len(required):
            session.status = TaskStatus.COMPLETED
        elif len(filled_required) > 0:
            session.status = TaskStatus.PARTIAL
        elif session.abandoned:
            session.status = TaskStatus.ABANDONED
        else:
            session.status = TaskStatus.FAILED

        return session

    def calculate_kappa_statistic(self,
                                  observed_agreement: float,
                                  expected_agreement: float) -> float:
        """
        Calculate Cohen's Kappa for information exchange
        
        κ = (P(A) - P(E)) / (1 - P(E))
        
        Args:
            observed_agreement: Actual agreement rate
            expected_agreement: Expected agreement by chance
            
        Returns:
            Kappa statistic
        """
        if expected_agreement >= 1.0:
            return 0.0

        kappa = (observed_agreement - expected_agreement) / (1 - expected_agreement)
        return max(0.0, min(1.0, kappa))

    def estimate_user_satisfaction(self,
                                   task_success: float,
                                   dialogue_cost: float,
                                   kappa: float) -> float:
        """
        Estimate user satisfaction using PARADISE framework
        
        Satisfaction = α·Success - β·Cost + γ·Kappa
        
        Args:
            task_success: Task completion score (0-1)
            dialogue_cost: Normalized dialogue length
            kappa: Information exchange quality
            
        Returns:
            Estimated satisfaction (0-1)
        """
        # PARADISE coefficients (from Walker et al., 1997)
        alpha = 0.5  # Weight for task success
        beta = 0.2   # Weight for efficiency (cost)
        gamma = 0.3  # Weight for information quality

        satisfaction = (alpha * task_success -
                       beta * dialogue_cost +
                       gamma * kappa)

        return max(0.0, min(1.0, satisfaction))

    def detect_false_containment(self, session: TaskSession) -> bool:
        """
        Detect false containment (user abandons without completion)
        
        Indicators:
        - Few turns (< 3)
        - Low slot filling rate (< 50%)
        - No explicit completion confirmation
        - Abrupt ending
        
        Args:
            session: Task session
            
        Returns:
            True if likely false containment
        """
        if session.status == TaskStatus.COMPLETED:
            return False

        # Check indicators
        num_turns = len(session.turns)
        required_slots = session.slots.get_required_slots()
        filled_slots = [s for s in required_slots if s.status == SlotStatus.FILLED]
        filling_rate = len(filled_slots) / len(required_slots) if required_slots else 0

        # Last turn analysis
        last_turn = session.turns[-1] if session.turns else None
        abrupt_ending = False
        if last_turn:
            # Check for frustration or abrupt endings
            frustration_keywords = ['nevermind', 'forget it', 'not working', 'bye', 'goodbye']
            abrupt_ending = any(kw in last_turn.text.lower() for kw in frustration_keywords)

        # False containment if:
        # 1. Short dialogue (< 3 turns)
        # 2. Low filling rate (< 50%)
        # 3. Abrupt ending OR marked as abandoned
        is_false_containment = (
            num_turns < 3 or
            (filling_rate < 0.5 and (abrupt_ending or session.abandoned))
        )

        return is_false_containment

    def evaluate_all_sessions(self,
                             sessions: List[TaskSession],
                             target_tsr: float = 0.85,
                             target_containment: float = 0.80
                             ) -> TaskCompletionMetrics:
        """
        Evaluate multiple sessions and calculate overall metrics
        
        Args:
            sessions: List of task sessions
            target_tsr: Target TSR for comparison (default 85%)
            target_containment: Target containment (default 80%)
            
        Returns:
            TaskCompletionMetrics with all results
        """
        print(f"\n{'='*70}")
        print("TASK COMPLETION EVALUATION")
        print(f"{'='*70}")
        print(f"Analyzing {len(sessions)} sessions...")

        # Process each session
        for session in sessions:
            self.evaluate_session(session)

        # Count task outcomes
        completed = len([s for s in sessions if s.status == TaskStatus.COMPLETED])
        partial = len([s for s in sessions if s.status == TaskStatus.PARTIAL])
        failed = len([s for s in sessions if s.status == TaskStatus.FAILED])
        abandoned = len([s for s in sessions if s.status == TaskStatus.ABANDONED])

        # Calculate TSR
        tsr = (completed / len(sessions)) * 100 if sessions else 0.0

        # Slot filling stats
        all_required_filled = 0
        all_required_total = 0
        all_optional_filled = 0
        all_optional_total = 0

        for session in sessions:
            required = session.slots.get_required_slots()
            optional = session.slots.get_optional_slots()

            all_required_total += len(required)
            all_required_filled += len([s for s in required if s.status == SlotStatus.FILLED])

            all_optional_total += len(optional)
            all_optional_filled += len([s for s in optional if s.status == SlotStatus.FILLED])

        slot_filling_rate = (all_required_filled / all_required_total * 100) if all_required_total > 0 else 0.0

        # Containment metrics
        escalated = len([s for s in sessions if s.escalated])
        containment_rate = ((len(sessions) - escalated) / len(sessions)) * 100 if sessions else 0.0

        # First-call resolution
        first_call = len([s for s in sessions if s.status == TaskStatus.COMPLETED and not s.escalated])
        fcr = (first_call / len(sessions)) * 100 if sessions else 0.0

        # False containment
        false_containment_count = len([s for s in sessions if self.detect_false_containment(s)])
        false_containment_rate = (false_containment_count / len(sessions)) * 100 if sessions else 0.0

        # Efficiency metrics
        completed_sessions = [s for s in sessions if s.status == TaskStatus.COMPLETED]
        avg_turns = np.mean([len(s.turns) for s in completed_sessions]) if completed_sessions else 0

        total_slots_filled = all_required_filled + all_optional_filled
        total_turns = sum(len(s.turns) for s in sessions)
        avg_slots_per_turn = total_slots_filled / total_turns if total_turns > 0 else 0

        # Slot correction rate
        total_changed = sum(len([s for s in session.slots.get_all_slots() if s.changed])
                          for session in sessions)
        slot_correction_rate = (total_changed / total_slots_filled) * 100 if total_slots_filled > 0 else 0

        # PARADISE framework metrics
        # Calculate average metrics across sessions
        kappas = []
        dialogue_costs = []
        satisfactions = []

        for session in sessions:
            # Observed agreement = slot filling rate
            required = session.slots.get_required_slots()
            filled = [s for s in required if s.status == SlotStatus.FILLED]
            observed = len(filled) / len(required) if required else 0

            # Expected agreement (baseline = random guessing)
            expected = 1.0 / len(required) if required else 0.5

            kappa = self.calculate_kappa_statistic(observed, expected)
            kappas.append(kappa)

            # Dialogue cost (normalized by turns)
            cost = min(len(session.turns) / 20.0, 1.0)  # Normalize to max 20 turns
            dialogue_costs.append(cost)

            # Task success
            success = 1.0 if session.status == TaskStatus.COMPLETED else 0.0

            # Satisfaction
            satisfaction = self.estimate_user_satisfaction(success, cost, kappa)
            satisfactions.append(satisfaction)

        avg_kappa = np.mean(kappas) if kappas else 0.0
        avg_dialogue_cost = np.mean(dialogue_costs) if dialogue_costs else 0.0
        avg_satisfaction = np.mean(satisfactions) if satisfactions else 0.0

        # Detailed slot breakdown
        slot_details = {}
        for slot_name in self.required_slots + self.optional_slots:
            filled_count = 0
            total_count = 0

            for session in sessions:
                slot = getattr(session.slots, slot_name, None)
                if slot:
                    total_count += 1
                    if slot.status == SlotStatus.FILLED:
                        filled_count += 1

            slot_details[slot_name] = {
                'filled': filled_count,
                'total': total_count,
                'fill_rate': (filled_count / total_count * 100) if total_count > 0 else 0
            }

        # Task breakdown
        task_breakdown = {
            'completed': completed,
            'partial': partial,
            'failed': failed,
            'abandoned': abandoned,
            'escalated': escalated,
            'false_containment': false_containment_count
        }

        metrics = TaskCompletionMetrics(
            tsr=tsr,
            total_tasks=len(sessions),
            completed_tasks=completed,
            partial_tasks=partial,
            failed_tasks=failed,
            abandoned_tasks=abandoned,
            required_slots_filled=all_required_filled,
            required_slots_total=all_required_total,
            optional_slots_filled=all_optional_filled,
            optional_slots_total=all_optional_total,
            slot_filling_rate=slot_filling_rate,
            kappa_statistic=avg_kappa,
            dialogue_success=completed / len(sessions) if sessions else 0,
            dialogue_cost=avg_dialogue_cost,
            user_satisfaction_estimate=avg_satisfaction,
            first_call_resolution=fcr,
            containment_rate=containment_rate,
            false_containment_rate=false_containment_rate,
            avg_turns_to_completion=avg_turns,
            avg_slots_per_turn=avg_slots_per_turn,
            slot_correction_rate=slot_correction_rate,
            slot_details=slot_details,
            task_breakdown=task_breakdown
        )

        # Print summary
        print(f"\n✓ Evaluation Complete:")
        print(f"  TSR: {tsr:.2f}% (target: {target_tsr*100:.0f}%)")
        print(f"  First-Call Resolution: {fcr:.2f}% (target: 85%)")
        print(f"  Containment: {containment_rate:.2f}% (target: {target_containment*100:.0f}%)")
        print(f"  False Containment: {false_containment_rate:.2f}%")
        print(f"  User Satisfaction (est): {avg_satisfaction:.2f}")

        return metrics


def format_single_session_report(slots: TravelAgentSlots, metrics: Dict[str, Any]) -> str:
    """Format a single session's task completion into readable report"""

    status = metrics.get('status', 'unknown')
    status_emoji = {
        'completed': '✅',
        'partial': '◐',
        'failed': '❌',
        'abandoned': '⊗'
    }.get(status, '?')

    required_pct = metrics.get('required_completion_rate', 0.0) * 100
    optional_pct = metrics.get('optional_completion_rate', 0.0) * 100

    report = f"""
╔══════════════════════════════════════════════════════════════════╗
║              TASK COMPLETION - SINGLE SESSION                     ║
╚══════════════════════════════════════════════════════════════════╝

📊 SESSION STATUS
──────────────────────────────────────────────────────────────────
  {status_emoji} Status: {status.upper()}
  Task Successful: {'Yes' if metrics.get('task_successful') else 'No'}
  User Abandoned: {'Yes' if metrics.get('user_abandoned') else 'No'}

🎯 SLOT FILLING
──────────────────────────────────────────────────────────────────
  Required Slots: {metrics.get('required_slots_filled', 0)}/{metrics.get('required_slots_total', 0)} ({required_pct:.1f}%)
  Optional Slots: {metrics.get('optional_slots_filled', 0)}/{metrics.get('optional_slots_total', 0)} ({optional_pct:.1f}%)

📋 EXTRACTED INFORMATION
──────────────────────────────────────────────────────────────────"""

    for slot in slots.get_all_slots():
        fill_status = "✓" if slot.status == SlotStatus.FILLED else "✗"
        req_label = "(required)" if slot.required else "(optional)"
        value_str = slot.value if slot.value else "—"
        
        # Add confidence if available
        conf_str = ""
        if slot.confidence > 0:
            conf_str = f" [conf: {slot.confidence:.2f}]"
        
        report += f"\n  {fill_status} {slot.name:20s} {req_label:12s}: {value_str}{conf_str}"

    if required_pct < 100:
        report += "\n\n⚠️  WARNING: Not all required slots filled - task cannot be completed"
    else:
        report += "\n\n✅ All required information collected - ready to proceed!"

    return report


def format_task_completion_report(metrics: TaskCompletionMetrics) -> str:
    """Format task completion metrics into readable report"""

    tsr_status = "✅" if metrics.tsr >= 85 else "⚠️" if metrics.tsr >= 70 else "❌"
    fcr_status = "✅" if metrics.first_call_resolution >= 85 else "⚠️" if metrics.first_call_resolution >= 70 else "❌"
    containment_status = "✅" if metrics.containment_rate >= 80 else "⚠️" if metrics.containment_rate >= 65 else "❌"

    report = f"""
╔══════════════════════════════════════════════════════════════════╗
║           TASK COMPLETION & SUCCESS RATE REPORT                   ║
╚══════════════════════════════════════════════════════════════════╝

📊 TASK SUCCESS RATE (TSR)
──────────────────────────────────────────────────────────────────
  {tsr_status} TSR:                      {metrics.tsr:.2f}%
  Total Tasks:              {metrics.total_tasks}
  ✓ Completed:              {metrics.completed_tasks} ({metrics.completed_tasks/metrics.total_tasks*100 if metrics.total_tasks else 0:.1f}%)
  ◐ Partial:                {metrics.partial_tasks} ({metrics.partial_tasks/metrics.total_tasks*100 if metrics.total_tasks else 0:.1f}%)
  ✗ Failed:                 {metrics.failed_tasks} ({metrics.failed_tasks/metrics.total_tasks*100 if metrics.total_tasks else 0:.1f}%)
  ⊗ Abandoned:              {metrics.abandoned_tasks} ({metrics.abandoned_tasks/metrics.total_tasks*100 if metrics.total_tasks else 0:.1f}%)

🎯 SLOT FILLING METRICS
──────────────────────────────────────────────────────────────────
  Required Slots Filled:    {metrics.required_slots_filled}/{metrics.required_slots_total} ({metrics.slot_filling_rate:.2f}%)
  Optional Slots Filled:    {metrics.optional_slots_filled}/{metrics.optional_slots_total}
  Avg Slots per Turn:       {metrics.avg_slots_per_turn:.2f}
  Slot Correction Rate:     {metrics.slot_correction_rate:.2f}%

📋 SLOT BREAKDOWN
──────────────────────────────────────────────────────────────────"""

    for slot_name, details in sorted(metrics.slot_details.items()):
        fill_status = "✓" if details['fill_rate'] >= 90 else "◐" if details['fill_rate'] >= 70 else "✗"
        report += f"\n  {fill_status} {slot_name:20s}: {details['filled']:>3}/{details['total']:<3} ({details['fill_rate']:>5.1f}%)"

    report += f"""

🎭 PARADISE FRAMEWORK
──────────────────────────────────────────────────────────────────
  Kappa Statistic (κ):      {metrics.kappa_statistic:.4f}
  Dialogue Success:         {metrics.dialogue_success:.4f}
  Dialogue Cost:            {metrics.dialogue_cost:.4f}
  User Satisfaction (est):  {metrics.user_satisfaction_estimate:.4f}

📞 CONTAINMENT METRICS
──────────────────────────────────────────────────────────────────
  {fcr_status} First-Call Resolution:  {metrics.first_call_resolution:.2f}% (target: 85%+)
  {containment_status} Containment Rate:       {metrics.containment_rate:.2f}% (target: 80%+)
  ⚠️  False Containment:      {metrics.false_containment_rate:.2f}%

⏱️  EFFICIENCY METRICS
──────────────────────────────────────────────────────────────────
  Avg Turns to Completion:  {metrics.avg_turns_to_completion:.2f}
  Avg Slots per Turn:       {metrics.avg_slots_per_turn:.2f}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📖 METRICS GUIDE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TSR (Task Success Rate)
  Formula: (Completed Tasks / Total Tasks) × 100
  Target: 85%+ for well-defined transactional flows

PARADISE Framework
  Kappa (κ): Information exchange quality
    Formula: (P(A) - P(E)) / (1 - P(E))
    Range: 0-1, higher is better

  User Satisfaction: α·Success - β·Cost + γ·Kappa
    Estimated using task success, dialogue efficiency, and info quality

First-Call Resolution (FCR)
  Tasks completed on first attempt without escalation
  Industry target: 85%+

Containment Rate
  Tasks handled without human escalation
  Industry target: 80%+

False Containment
  Users abandoning without completion
  High containment + low CSAT = false containment
  Warning sign: Users giving up rather than being helped

Slot Filling Rate
  Percentage of required information captured
  100% required for task completion
"""

    # Warnings section
    warnings_text = []

    if metrics.tsr < 70:
        warnings_text.append("⚠️  TSR below 70% - critical issue")
    if metrics.false_containment_rate > 20:
        warnings_text.append("⚠️  High false containment - users abandoning")
    if metrics.containment_rate < 65:
        warnings_text.append("⚠️  Low containment - high escalation rate")
    if metrics.user_satisfaction_estimate < 0.6:
        warnings_text.append("⚠️  Low estimated user satisfaction")

    if warnings_text:
        report += "\n⚠️  WARNINGS\n"
        report += "──────────────────────────────────────────────────────────────────\n"
        for warning in warnings_text:
            report += f"  {warning}\n"

    return report


def main():
    """Example usage - supports both single transcript and batch evaluation"""
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  Single transcript:  python task_completion.py <transcript.txt>")
        print("  Batch sessions:     python task_completion.py --batch <sessions.json>")
        sys.exit(1)

    evaluator = TaskCompletionEvaluator()

    # Batch mode
    if sys.argv[1] == '--batch':
        if len(sys.argv) < 3:
            print("Error: --batch requires sessions JSON file")
            sys.exit(1)

        # Load sessions from JSON
        with open(sys.argv[2], 'r') as f:
            sessions_data = json.load(f)

        # Convert to TaskSession objects
        sessions = []
        for sess_data in sessions_data:
            # Create session (simplified - adapt to your data format)
            session = TaskSession(
                session_id=sess_data['id'],
                turns=[],
                slots=TravelAgentSlots(),
                status=TaskStatus.PARTIAL
            )
            # Add turns, etc.
            sessions.append(session)

        # Evaluate
        metrics = evaluator.evaluate_all_sessions(sessions)

        # Print report
        print(format_task_completion_report(metrics))

        # Save results
        with open('task_completion_results.json', 'w') as f:
            json.dump(asdict(metrics), f, indent=2)

    # Single transcript mode
    else:
        transcript_path = sys.argv[1]

        # Read transcript
        with open(transcript_path, 'r') as f:
            transcript = f.read()

        print(f"\nEvaluating transcript: {transcript_path}")
        print("="*70)

        # Evaluate
        slots, metrics = evaluator.evaluate_transcript(transcript, verbose=True)

        # Print report
        print(format_single_session_report(slots, metrics))

        # Save results
        output_path = transcript_path.replace('.txt', '_task_completion.json')
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)

        print(f"\n💾 Results saved to: {output_path}")


if __name__ == "__main__":
    main()