import itertools
import random
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

st.set_page_config(
    page_title="Blackjack Monte Carlo",
    page_icon="🃏",
    layout="wide",
)

# ============================================================
# CONFIGURAZIONE
# ============================================================
BLACKJACK_PAYOUT = 1.5
DEFAULT_SIMULATIONS = 5000
DEFAULT_DECKS = 6
DEFAULT_SEED = 42
MAX_HIT_RECURSION_DEPTH = 3
INITIAL_BANKROLL = 1000.0
DEFAULT_BET = 10.0

RANKS = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]
UPCARD_OPTIONS = ["Ignora"] + RANKS

CARD_VALUES = {
    "A": 11,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "10": 10,
    "J": 10,
    "Q": 10,
    "K": 10,
}

COPIES_PER_DECK = {rank: 4 for rank in RANKS}


# ============================================================
# RANDOM SEED PER GIOCO ATTIVO
# ============================================================
def generate_live_seed() -> int:
    return int(time.time_ns() % 1_000_000_000) + random.randint(1, 999999)


# ============================================================
# UTILITÀ BASE
# ============================================================
def fresh_shoe_counter(n_decks: int) -> Counter:
    shoe = Counter()
    for rank, copies in COPIES_PER_DECK.items():
        shoe[rank] = copies * n_decks
    return shoe


def fresh_shoe_list(n_decks: int, rng: random.Random) -> List[str]:
    cards: List[str] = []
    for rank in RANKS:
        cards.extend([rank] * (4 * n_decks))
    rng.shuffle(cards)
    return cards


def remove_known_cards(shoe: Counter, cards: List[str]) -> Counter:
    updated = shoe.copy()
    for card in cards:
        if updated[card] <= 0:
            raise ValueError(f"Carte insufficienti nel sabot per rimuovere {card}")
        updated[card] -= 1
        if updated[card] == 0:
            del updated[card]
    return updated


def draw_random_card(shoe: Counter, rng: random.Random) -> str:
    total = sum(shoe.values())
    if total <= 0:
        raise RuntimeError("Sabot vuoto")

    pick = rng.randint(1, total)
    running = 0
    for card, count in shoe.items():
        running += count
        if pick <= running:
            shoe[card] -= 1
            if shoe[card] == 0:
                del shoe[card]
            return card
    raise RuntimeError("Errore nel pescaggio casuale")


def hand_total(cards: List[str]) -> Tuple[int, bool]:
    total = sum(CARD_VALUES[c] for c in cards)
    aces_as_11 = sum(1 for c in cards if c == "A")

    while total > 21 and aces_as_11 > 0:
        total -= 10
        aces_as_11 -= 1

    soft = any(c == "A" for c in cards) and aces_as_11 > 0 and total <= 21
    return total, soft


def is_blackjack(cards: List[str]) -> bool:
    total, _ = hand_total(cards)
    return len(cards) == 2 and total == 21


def is_bust(cards: List[str]) -> bool:
    total, _ = hand_total(cards)
    return total > 21


def same_value(card1: str, card2: str) -> bool:
    return CARD_VALUES[card1] == CARD_VALUES[card2]


def can_split(cards: List[str]) -> bool:
    return len(cards) == 2 and same_value(cards[0], cards[1])


def cards_to_string(cards: List[str]) -> str:
    return ",".join(cards)


def hand_label(cards: List[str]) -> str:
    c1, c2 = cards[0], cards[1]
    total, soft = hand_total(cards[:2])
    split_text = " | split" if can_split(cards[:2]) else ""

    if c1 == c2:
        return f"{c1},{c2} (tot {total}){split_text}"
    if "A" in cards[:2]:
        return f"{c1},{c2} (soft {total}){split_text}"
    return f"{c1},{c2} (hard {total}){split_text}"


def full_hand_label(cards: List[str]) -> str:
    total, soft = hand_total(cards)
    kind = "soft" if soft else "hard"
    if total > 21:
        kind = "bust"
    return f"{' '.join(cards)} → {total} ({kind})"


def dealer_should_hit(cards: List[str], dealer_hits_soft_17: bool) -> bool:
    total, soft = hand_total(cards)
    if total < 17:
        return True
    if total > 17:
        return False
    return dealer_hits_soft_17 and soft


def play_out_dealer(
    shoe: Counter,
    rng: random.Random,
    dealer_hits_soft_17: bool,
    dealer_upcard: Optional[str] = None,
) -> List[str]:
    if dealer_upcard is None:
        dealer_cards = [draw_random_card(shoe, rng), draw_random_card(shoe, rng)]
    else:
        dealer_cards = [dealer_upcard, draw_random_card(shoe, rng)]

    while dealer_should_hit(dealer_cards, dealer_hits_soft_17):
        dealer_cards.append(draw_random_card(shoe, rng))
    return dealer_cards


def settle(player_cards: List[str], dealer_cards: List[str], natural_blackjack_pays_3_to_2: bool = True) -> float:
    player_total, _ = hand_total(player_cards)
    dealer_total, _ = hand_total(dealer_cards)

    if player_total > 21:
        return -1.0

    if dealer_total > 21:
        if natural_blackjack_pays_3_to_2 and is_blackjack(player_cards):
            return BLACKJACK_PAYOUT
        return 1.0

    player_bj = natural_blackjack_pays_3_to_2 and is_blackjack(player_cards)
    dealer_bj = is_blackjack(dealer_cards)

    if player_bj and not dealer_bj:
        return BLACKJACK_PAYOUT
    if dealer_bj and not player_bj:
        return -1.0
    if player_total > dealer_total:
        return 1.0
    if player_total < dealer_total:
        return -1.0
    return 0.0


# ============================================================
# GRAFICA TAVOLO E CARTE CON PIL
# ============================================================
def get_card_color(rank: str) -> str:
    red_ranks = {"2", "4", "6", "8", "10", "Q"}
    return "red" if rank in red_ranks else "black"


def get_card_suit(rank: str) -> str:
    suit_map = {
        "A": "♠",
        "2": "♥",
        "3": "♣",
        "4": "♦",
        "5": "♠",
        "6": "♥",
        "7": "♣",
        "8": "♦",
        "9": "♠",
        "10": "♥",
        "J": "♣",
        "Q": "♦",
        "K": "♠",
    }
    return suit_map.get(rank, "♠")


def load_font(size: int):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    return ImageFont.load_default()


def draw_card_image(draw: ImageDraw.ImageDraw, x: int, y: int, w: int, h: int, rank: str, hidden: bool = False):
    if hidden:
        draw.rounded_rectangle((x, y, x + w, y + h), radius=14, fill="#1e3a8a", outline="white", width=3)
        for i in range(x + 10, x + w - 10, 12):
            draw.line((i, y + 10, i, y + h - 10), fill="#93c5fd", width=1)
        for j in range(y + 10, y + h - 10, 12):
            draw.line((x + 10, j, x + w - 10, j), fill="#60a5fa", width=1)
        return

    draw.rounded_rectangle((x, y, x + w, y + h), radius=14, fill="white", outline="#d1d5db", width=3)

    color = "#c62828" if get_card_color(rank) == "red" else "#111111"
    suit = get_card_suit(rank)

    small_font = load_font(26)
    center_font = load_font(48)

    draw.text((x + 12, y + 10), f"{rank}", fill=color, font=small_font)
    draw.text((x + 14, y + 42), suit, fill=color, font=small_font)

    bbox = draw.textbbox((0, 0), suit, font=center_font)
    sw = bbox[2] - bbox[0]
    sh = bbox[3] - bbox[1]
    draw.text((x + (w - sw) / 2, y + (h - sh) / 2 - 10), suit, fill=color, font=center_font)

    draw.text((x + w - 30, y + h - 54), suit, fill=color, font=small_font)
    draw.text((x + w - 40, y + h - 26), f"{rank}", fill=color, font=small_font)


def build_blackjack_table_image(
    dealer_cards: List[str],
    player_cards: List[str],
    round_over: bool,
    player_bet: float,
    bankroll: float,
    result_text: str,
    payout_units: float,
):
    width, height = 1500, 900
    img = Image.new("RGB", (width, height), "#0b3d1f")
    draw = ImageDraw.Draw(img)

    title_font = load_font(42)
    label_font = load_font(34)
    text_font = load_font(26)
    small_font = load_font(21)

    # Tavolo
    draw.rounded_rectangle(
        (25, 25, width - 25, height - 25),
        radius=55,
        fill="#146c3b",
        outline="#6b3f1d",
        width=16
    )
    draw.arc((140, 110, width - 140, height - 110), start=200, end=340, fill="#d6c48a", width=6)

    # Titolo
    title = "BLACKJACK TABLE"
    tb = draw.textbbox((0, 0), title, font=title_font)
    draw.text(((width - (tb[2] - tb[0])) / 2, 40), title, fill="#f3e6b3", font=title_font)

    # Info tavolo
    info = f"Bankroll: € {bankroll:,.2f}    Puntata: € {player_bet:.2f}"
    draw.text((55, 105), info, fill="white", font=text_font)

    # ZONA BANCO
    draw.text((100, 185), "Banco", fill="white", font=label_font)

    dealer_card_y = 170
    dealer_card_x = 470   # spostato più a destra
    card_w, card_h = 110, 160
    card_gap = 130

    if dealer_cards:
        for i, c in enumerate(dealer_cards):
            draw_card_image(
                draw,
                dealer_card_x + i * card_gap,
                dealer_card_y,
                card_w,
                card_h,
                c,
                hidden=(i == 1 and not round_over),
            )

        if round_over:
            dealer_total, _ = hand_total(dealer_cards)
            draw.text((100, 235), f"Totale banco: {dealer_total}", fill="#eef7ee", font=text_font)
        else:
            visible_total = CARD_VALUES[dealer_cards[0]] if dealer_cards[0] != "A" else 11
            draw.text((100, 235), f"Carta visibile: {visible_total}", fill="#eef7ee", font=text_font)
    else:
        draw.text((dealer_card_x, 235), "Nessuna carta distribuita", fill="#eef7ee", font=text_font)

    # ZONA GIOCATORE
    draw.text((100, 560), "Giocatore", fill="white", font=label_font)

    player_card_y = 545
    player_card_x = 470   # spostato più a destra

    if player_cards:
        for i, c in enumerate(player_cards):
            draw_card_image(
                draw,
                player_card_x + i * card_gap,
                player_card_y,
                card_w,
                card_h,
                c,
                hidden=False,
            )

        player_total, _ = hand_total(player_cards)
        draw.text((100, 615), f"Totale giocatore: {player_total}", fill="#eef7ee", font=text_font)
    else:
        draw.text((player_card_x, 615), "Premi Distribuisci per iniziare", fill="#eef7ee", font=text_font)

    # BANNER RISULTATO
    if result_text:
        if payout_units < 0:
            fill = "#7f1d1d"
            outline = "#ef4444"
        elif payout_units == 0:
            fill = "#78350f"
            outline = "#f59e0b"
        else:
            fill = "#14532d"
            outline = "#22c55e"

        bx1, by1, bx2, by2 = 980, 700, 1420, 790
        draw.rounded_rectangle((bx1, by1, bx2, by2), radius=18, fill=fill, outline=outline, width=4)

        result_msg = f"Esito: {result_text}"
        payout_msg = f"Payout unità: {payout_units:+.2f}"

        draw.text((bx1 + 20, by1 + 16), result_msg, fill="white", font=text_font)
        draw.text((bx1 + 20, by1 + 48), payout_msg, fill="white", font=small_font)

    return img


# ============================================================
# STRUTTURE DATI
# ============================================================
@dataclass
class MoveResult:
    move: str
    ev: float
    win_prob: float
    push_prob: float
    loss_prob: float


@dataclass
class ThirdCardOutcome:
    third_card: str
    frequency: int
    probability: float
    resulting_hand: str
    total: int
    soft: bool
    bust: bool
    win_prob_after_third: float
    push_prob_after_third: float
    loss_prob_after_third: float
    ev_after_third: float


@dataclass
class LiveHand:
    cards: List[str] = field(default_factory=list)
    bet: float = DEFAULT_BET
    finished: bool = False
    surrendered: bool = False
    doubled: bool = False
    result_text: str = ""
    payout_units: float = 0.0
    one_hit_used: bool = False


@dataclass
class LiveGameState:
    bankroll: float = INITIAL_BANKROLL
    shoe: List[str] = field(default_factory=list)
    dealer_cards: List[str] = field(default_factory=list)
    player_hand: LiveHand = field(default_factory=LiveHand)
    in_round: bool = False
    round_over: bool = False
    message: str = ""
    n_decks: int = DEFAULT_DECKS
    seed: int = DEFAULT_SEED
    frozen_pre_move: Optional[Dict[str, float]] = None


# ============================================================
# MOTORE MONTE CARLO
# ============================================================
class BlackjackStudy:
    def __init__(
        self,
        simulations: int,
        n_decks: int = DEFAULT_DECKS,
        seed: int = DEFAULT_SEED,
        dealer_hits_soft_17: bool = False,
        allow_surrender: bool = True,
        allow_double: bool = True,
        allow_split: bool = True,
        dealer_upcard: Optional[str] = None,
    ):
        self.simulations = simulations
        self.n_decks = n_decks
        self.seed = seed
        self.dealer_hits_soft_17 = dealer_hits_soft_17
        self.allow_surrender = allow_surrender
        self.allow_double = allow_double
        self.allow_split = allow_split
        self.dealer_upcard = dealer_upcard

    def _rng(self, offset: int) -> random.Random:
        return random.Random(self.seed + offset)

    def _base_shoe_for_player(self, player_cards: List[str]) -> Counter:
        shoe = fresh_shoe_counter(self.n_decks)
        known = player_cards[:]
        if self.dealer_upcard is not None:
            known.append(self.dealer_upcard)
        return remove_known_cards(shoe, known)

    def continue_after_hit_best_win_prob(
        self,
        player_cards: List[str],
        shoe: Counter,
        rng: random.Random,
        depth: int = 0,
    ) -> float:
        total, _ = hand_total(player_cards)
        if total > 21:
            return -1.0

        if depth >= MAX_HIT_RECURSION_DEPTH:
            dealer = play_out_dealer(
                shoe,
                rng,
                self.dealer_hits_soft_17,
                dealer_upcard=self.dealer_upcard,
            )
            return settle(player_cards, dealer, natural_blackjack_pays_3_to_2=False)

        stand_shoe = shoe.copy()
        dealer_stand = play_out_dealer(
            stand_shoe,
            rng,
            self.dealer_hits_soft_17,
            dealer_upcard=self.dealer_upcard,
        )
        stand_result = settle(player_cards, dealer_stand, natural_blackjack_pays_3_to_2=False)

        hit_shoe = shoe.copy()
        next_card = draw_random_card(hit_shoe, rng)
        new_player = player_cards[:] + [next_card]
        if is_bust(new_player):
            hit_result = -1.0
        else:
            hit_result = self.continue_after_hit_best_win_prob(new_player, hit_shoe, rng, depth + 1)

        return max(stand_result, hit_result)

    def evaluate_existing_hand(self, player_cards: List[str], shoe: Counter) -> MoveResult:
        rng = self._rng(7000)
        wins = pushes = losses = 0
        total_ev = 0.0

        for _ in range(self.simulations):
            sim_shoe = shoe.copy()
            total, _ = hand_total(player_cards)

            if total > 21:
                ev = -1.0
            else:
                stand_shoe = sim_shoe.copy()
                dealer1 = play_out_dealer(
                    stand_shoe,
                    rng,
                    self.dealer_hits_soft_17,
                    dealer_upcard=self.dealer_upcard,
                )
                stand_ev = settle(player_cards, dealer1, natural_blackjack_pays_3_to_2=False)

                if len(sim_shoe) == 0:
                    hit_ev = -1.0
                else:
                    hit_shoe = sim_shoe.copy()
                    next_card = draw_random_card(hit_shoe, rng)
                    new_player = player_cards[:] + [next_card]
                    if is_bust(new_player):
                        hit_ev = -1.0
                    else:
                        hit_ev = self.continue_after_hit_best_win_prob(new_player, hit_shoe, rng, depth=1)

                ev = max(stand_ev, hit_ev)

            total_ev += ev
            if ev > 0:
                wins += 1
            elif ev < 0:
                losses += 1
            else:
                pushes += 1

        return MoveResult(
            move="POST_HIT_OPTIMAL",
            ev=total_ev / self.simulations,
            win_prob=wins / self.simulations,
            push_prob=pushes / self.simulations,
            loss_prob=losses / self.simulations,
        )

    def evaluate_stand(self, player_cards: List[str], shoe: Counter) -> MoveResult:
        wins = pushes = losses = 0
        total_ev = 0.0
        rng = self._rng(1000)

        for _ in range(self.simulations):
            sim_shoe = shoe.copy()
            dealer = play_out_dealer(
                sim_shoe,
                rng,
                self.dealer_hits_soft_17,
                dealer_upcard=self.dealer_upcard,
            )
            ev = settle(player_cards, dealer, natural_blackjack_pays_3_to_2=True)
            total_ev += ev

            if ev > 0:
                wins += 1
            elif ev < 0:
                losses += 1
            else:
                pushes += 1

        return MoveResult("STAI", total_ev / self.simulations, wins / self.simulations, pushes / self.simulations, losses / self.simulations)

    def evaluate_hit(self, player_cards: List[str], shoe: Counter) -> MoveResult:
        wins = pushes = losses = 0
        total_ev = 0.0
        rng = self._rng(2000)

        for _ in range(self.simulations):
            sim_shoe = shoe.copy()
            player = player_cards[:] + [draw_random_card(sim_shoe, rng)]

            if is_bust(player):
                losses += 1
                total_ev += -1.0
                continue

            final_ev = self.continue_after_hit_best_win_prob(player, sim_shoe, rng, depth=0)
            total_ev += final_ev

            if final_ev > 0:
                wins += 1
            elif final_ev < 0:
                losses += 1
            else:
                pushes += 1

        return MoveResult("CARTA", total_ev / self.simulations, wins / self.simulations, pushes / self.simulations, losses / self.simulations)

    def evaluate_double(self, player_cards: List[str], shoe: Counter) -> MoveResult:
        wins = pushes = losses = 0
        total_ev = 0.0
        rng = self._rng(3000)

        for _ in range(self.simulations):
            sim_shoe = shoe.copy()
            player = player_cards[:] + [draw_random_card(sim_shoe, rng)]

            if is_bust(player):
                losses += 1
                total_ev += -2.0
                continue

            dealer = play_out_dealer(
                sim_shoe,
                rng,
                self.dealer_hits_soft_17,
                dealer_upcard=self.dealer_upcard,
            )
            base_ev = settle(player, dealer, natural_blackjack_pays_3_to_2=False)
            ev = 2.0 * base_ev
            total_ev += ev

            if base_ev > 0:
                wins += 1
            elif base_ev < 0:
                losses += 1
            else:
                pushes += 1

        return MoveResult("RADDOPPIA", total_ev / self.simulations, wins / self.simulations, pushes / self.simulations, losses / self.simulations)

    def evaluate_surrender(self) -> MoveResult:
        return MoveResult("RESA", -0.5, 0.0, 0.0, 1.0)

    def _play_split_hand_follow_best_win_prob(
        self,
        player_cards: List[str],
        shoe: Counter,
        rng: random.Random,
    ) -> float:
        if player_cards[0] == "A":
            dealer = play_out_dealer(
                shoe,
                rng,
                self.dealer_hits_soft_17,
                dealer_upcard=self.dealer_upcard,
            )
            return settle(player_cards, dealer, natural_blackjack_pays_3_to_2=False)

        stand_shoe = shoe.copy()
        dealer1 = play_out_dealer(
            stand_shoe,
            rng,
            self.dealer_hits_soft_17,
            dealer_upcard=self.dealer_upcard,
        )
        stand_ev = settle(player_cards, dealer1, natural_blackjack_pays_3_to_2=False)

        hit_shoe = shoe.copy()
        player_hit = player_cards[:] + [draw_random_card(hit_shoe, rng)]
        if is_bust(player_hit):
            hit_ev = -1.0
        else:
            hit_ev = self.continue_after_hit_best_win_prob(player_hit, hit_shoe, rng, depth=0)

        options = [("STAI", stand_ev), ("CARTA", hit_ev)]
        if self.allow_double:
            double_shoe = shoe.copy()
            player_double = player_cards[:] + [draw_random_card(double_shoe, rng)]
            if is_bust(player_double):
                double_ev = -2.0
            else:
                dealer3 = play_out_dealer(
                    double_shoe,
                    rng,
                    self.dealer_hits_soft_17,
                    dealer_upcard=self.dealer_upcard,
                )
                double_ev = 2.0 * settle(player_double, dealer3, natural_blackjack_pays_3_to_2=False)
            options.append(("RADDOPPIA", double_ev))

        _, best_ev = max(options, key=lambda x: x[1])
        return best_ev

    def evaluate_split(self, player_cards: List[str], shoe: Counter) -> Optional[MoveResult]:
        if not can_split(player_cards):
            return None

        wins = pushes = losses = 0
        total_ev = 0.0
        rng = self._rng(4000)

        c1, c2 = player_cards

        for _ in range(self.simulations):
            sim_shoe = shoe.copy()

            hand1 = [c1, draw_random_card(sim_shoe, rng)]
            hand2 = [c2, draw_random_card(sim_shoe, rng)]

            ev1 = self._play_split_hand_follow_best_win_prob(hand1, sim_shoe.copy(), rng)
            ev2 = self._play_split_hand_follow_best_win_prob(hand2, sim_shoe.copy(), rng)

            combined_ev = ev1 + ev2
            total_ev += combined_ev

            if combined_ev > 0:
                wins += 1
            elif combined_ev < 0:
                losses += 1
            else:
                pushes += 1

        return MoveResult("SPLIT", total_ev / self.simulations, wins / self.simulations, pushes / self.simulations, losses / self.simulations)

    def analyze_starting_hand(self, player_cards: List[str]) -> Dict[str, MoveResult]:
        shoe = self._base_shoe_for_player(player_cards)

        results: Dict[str, MoveResult] = {
            "STAI": self.evaluate_stand(player_cards, shoe),
            "CARTA": self.evaluate_hit(player_cards, shoe),
        }

        if self.allow_double and len(player_cards) == 2:
            results["RADDOPPIA"] = self.evaluate_double(player_cards, shoe)

        if self.allow_surrender and len(player_cards) == 2:
            results["RESA"] = self.evaluate_surrender()

        if self.allow_split and len(player_cards) == 2 and can_split(player_cards):
            split_result = self.evaluate_split(player_cards, shoe)
            if split_result is not None:
                results["SPLIT"] = split_result

        return results

    def recommended_move_by_win_prob(self, player_cards: List[str]) -> Tuple[MoveResult, Dict[str, MoveResult]]:
        results = self.analyze_starting_hand(player_cards)
        best = max(results.values(), key=lambda r: r.win_prob)
        return best, results

    def best_move_by_ev(self, results: Dict[str, MoveResult]) -> MoveResult:
        return max(results.values(), key=lambda r: r.ev)

    def simulate_recommended_move(self, player_cards: List[str], move: str) -> MoveResult:
        shoe = self._base_shoe_for_player(player_cards)

        if move == "STAI":
            return self.evaluate_stand(player_cards, shoe)
        if move == "CARTA":
            return self.evaluate_hit(player_cards, shoe)
        if move == "RADDOPPIA":
            return self.evaluate_double(player_cards, shoe)
        if move == "RESA":
            return self.evaluate_surrender()
        if move == "SPLIT":
            split_result = self.evaluate_split(player_cards, shoe)
            if split_result is None:
                raise ValueError("Split non consentito per questa mano")
            return split_result

        raise ValueError(f"Mossa non riconosciuta: {move}")

    def third_card_breakdown(self, player_cards: List[str]) -> List[ThirdCardOutcome]:
        shoe = self._base_shoe_for_player(player_cards)

        total_remaining = sum(shoe.values())
        outcomes: List[ThirdCardOutcome] = []

        for third_card in RANKS:
            freq = shoe.get(third_card, 0)
            if freq == 0:
                continue

            prob = freq / total_remaining
            updated_shoe = shoe.copy()
            updated_shoe[third_card] -= 1
            if updated_shoe[third_card] == 0:
                del updated_shoe[third_card]

            new_hand = player_cards[:] + [third_card]
            total, soft = hand_total(new_hand)
            bust = total > 21

            if bust:
                result = MoveResult("POST_HIT_BUST", -1.0, 0.0, 0.0, 1.0)
            else:
                result = self.evaluate_existing_hand(new_hand, updated_shoe)

            outcomes.append(
                ThirdCardOutcome(
                    third_card=third_card,
                    frequency=freq,
                    probability=prob,
                    resulting_hand=cards_to_string(new_hand),
                    total=total,
                    soft=soft,
                    bust=bust,
                    win_prob_after_third=result.win_prob,
                    push_prob_after_third=result.push_prob,
                    loss_prob_after_third=result.loss_prob,
                    ev_after_third=result.ev,
                )
            )

        outcomes.sort(key=lambda x: (-x.probability, x.third_card))
        return outcomes


# ============================================================
# CACHE FUNZIONI STUDIO
# ============================================================
def normalize_upcard(upcard_choice: str) -> Optional[str]:
    return None if upcard_choice == "Ignora" else upcard_choice


def format_dealer_upcard_label(upcard_choice: str) -> str:
    return "ignota" if upcard_choice == "Ignora" else upcard_choice


def all_starting_rank_pairs() -> List[Tuple[str, str]]:
    return list(itertools.combinations_with_replacement(RANKS, 2))


def format_details(results: Dict[str, MoveResult]) -> str:
    ordered = ["STAI", "CARTA", "RADDOPPIA", "RESA", "SPLIT"]
    out = []
    for key in ordered:
        if key in results:
            r = results[key]
            out.append(
                f"{key}: win={r.win_prob:.3f}, push={r.push_prob:.3f}, "
                f"loss={r.loss_prob:.3f}, ev={r.ev:.3f}"
            )
    return " | ".join(out)


@st.cache_data(show_spinner=False)
def run_study(
    simulations: int,
    n_decks: int,
    dealer_hits_soft_17: bool,
    allow_surrender: bool,
    allow_double: bool,
    allow_split: bool,
    seed: int,
    dealer_upcard_choice: str,
) -> pd.DataFrame:
    engine = BlackjackStudy(
        simulations=simulations,
        n_decks=n_decks,
        seed=seed,
        dealer_hits_soft_17=dealer_hits_soft_17,
        allow_surrender=allow_surrender,
        allow_double=allow_double,
        allow_split=allow_split,
        dealer_upcard=normalize_upcard(dealer_upcard_choice),
    )

    rows = []
    for pair in all_starting_rank_pairs():
        player_cards = [pair[0], pair[1]]
        total, soft = hand_total(player_cards)

        recommended_result, all_results = engine.recommended_move_by_win_prob(player_cards)
        best_ev_move = engine.best_move_by_ev(all_results)

        rows.append(
            {
                "Carta banco scoperta": format_dealer_upcard_label(dealer_upcard_choice),
                "Carta 1": pair[0],
                "Carta 2": pair[1],
                "Mano": hand_label(player_cards),
                "Totale iniziale": total,
                "Soft": soft,
                "Split consentito": can_split(player_cards),
                "Mossa consigliata": recommended_result.move,
                "Prob. vittoria prima della mossa": recommended_result.win_prob,
                "Prob. pareggio prima della mossa": recommended_result.push_prob,
                "Prob. sconfitta prima della mossa": recommended_result.loss_prob,
                "EV prima della mossa": recommended_result.ev,
                "Mossa migliore per EV": best_ev_move.move,
                "EV migliore assoluto": best_ev_move.ev,
                "Dettagli tutte le mosse": format_details(all_results),
            }
        )

    df = pd.DataFrame(rows)

    percent_cols = [
        "Prob. vittoria prima della mossa",
        "Prob. pareggio prima della mossa",
        "Prob. sconfitta prima della mossa",
    ]
    for col in percent_cols:
        df[f"{col} %"] = (df[col] * 100).round(2)

    df["EV prima della mossa"] = df["EV prima della mossa"].round(3)
    df["EV migliore assoluto"] = df["EV migliore assoluto"].round(3)
    return df


@st.cache_data(show_spinner=False)
def run_third_card_breakdown(
    simulations: int,
    n_decks: int,
    dealer_hits_soft_17: bool,
    allow_surrender: bool,
    allow_double: bool,
    allow_split: bool,
    seed: int,
    card1: str,
    card2: str,
    dealer_upcard_choice: str,
) -> pd.DataFrame:
    engine = BlackjackStudy(
        simulations=simulations,
        n_decks=n_decks,
        seed=seed,
        dealer_hits_soft_17=dealer_hits_soft_17,
        allow_surrender=allow_surrender,
        allow_double=allow_double,
        allow_split=allow_split,
        dealer_upcard=normalize_upcard(dealer_upcard_choice),
    )

    outcomes = engine.third_card_breakdown([card1, card2])

    rows = []
    for item in outcomes:
        rows.append(
            {
                "Carta banco scoperta": format_dealer_upcard_label(dealer_upcard_choice),
                "Terza carta": item.third_card,
                "Prob. estrazione": round(item.probability * 100, 2),
                "Frequenza residua": item.frequency,
                "Nuova mano": item.resulting_hand,
                "Totale": item.total,
                "Soft": item.soft,
                "Bust": item.bust,
                "Prob. vittoria dopo terza carta %": round(item.win_prob_after_third * 100, 2),
                "Prob. pareggio dopo terza carta %": round(item.push_prob_after_third * 100, 2),
                "Prob. sconfitta dopo terza carta %": round(item.loss_prob_after_third * 100, 2),
                "EV dopo terza carta": round(item.ev_after_third, 3),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# LIVE GAME UTILITIES
# ============================================================
def ensure_live_state():
    if "live_game" not in st.session_state:
        live_seed = generate_live_seed()
        rng = random.Random(live_seed)
        st.session_state.live_game = LiveGameState(
            bankroll=INITIAL_BANKROLL,
            shoe=fresh_shoe_list(DEFAULT_DECKS, rng),
            dealer_cards=[],
            player_hand=LiveHand(cards=[], bet=DEFAULT_BET),
            in_round=False,
            round_over=False,
            message="",
            n_decks=DEFAULT_DECKS,
            seed=live_seed,
            frozen_pre_move=None,
        )


def live_game() -> LiveGameState:
    ensure_live_state()
    return st.session_state.live_game


def reset_live_shoe(n_decks: int, seed: Optional[int] = None):
    live_seed = generate_live_seed() if seed is None else seed
    rng = random.Random(live_seed)
    g = live_game()
    g.n_decks = n_decks
    g.seed = live_seed
    g.shoe = fresh_shoe_list(n_decks, rng)


def draw_from_live_shoe() -> str:
    g = live_game()
    if len(g.shoe) < 30:
        reset_live_shoe(g.n_decks)
    return g.shoe.pop()


def freeze_initial_probabilities(
    player_cards: List[str],
    dealer_upcard: str,
    simulations: int,
    n_decks: int,
    dealer_hits_soft_17: bool,
    allow_surrender: bool,
    allow_double: bool,
    allow_split: bool,
    seed: int,
):
    engine = BlackjackStudy(
        simulations=simulations,
        n_decks=n_decks,
        seed=seed,
        dealer_hits_soft_17=dealer_hits_soft_17,
        allow_surrender=allow_surrender and len(player_cards) == 2,
        allow_double=allow_double and len(player_cards) == 2,
        allow_split=allow_split and len(player_cards) == 2,
        dealer_upcard=dealer_upcard,
    )
    recommended, all_results = engine.recommended_move_by_win_prob(player_cards)
    best_ev = engine.best_move_by_ev(all_results)

    live_game().frozen_pre_move = {
        "move": recommended.move,
        "win": recommended.win_prob,
        "push": recommended.push_prob,
        "loss": recommended.loss_prob,
        "ev": recommended.ev,
        "best_ev_move": best_ev.move,
        "details": format_details(all_results),
        "dealer_upcard": dealer_upcard,
    }


def start_live_round(
    bet: float,
    simulations: int,
    n_decks: int,
    dealer_hits_soft_17: bool,
    allow_surrender: bool,
    allow_double: bool,
    allow_split: bool,
    seed: int,
):
    g = live_game()
    if g.in_round and not g.round_over:
        return
    if bet <= 0 or bet > g.bankroll:
        g.message = "Puntata non valida."
        return

    g.bankroll -= bet
    g.player_hand = LiveHand(cards=[draw_from_live_shoe(), draw_from_live_shoe()], bet=bet)
    g.dealer_cards = [draw_from_live_shoe(), draw_from_live_shoe()]
    g.in_round = True
    g.round_over = False
    g.message = "Mano iniziata."
    g.frozen_pre_move = None

    freeze_initial_probabilities(
        player_cards=g.player_hand.cards,
        dealer_upcard=g.dealer_cards[0],
        simulations=simulations,
        n_decks=n_decks,
        dealer_hits_soft_17=dealer_hits_soft_17,
        allow_surrender=allow_surrender,
        allow_double=allow_double,
        allow_split=allow_split,
        seed=seed,
    )

    if is_blackjack(g.player_hand.cards) or is_blackjack(g.dealer_cards):
        finish_live_round()


def dealer_play_live(dealer_hits_soft_17: bool):
    g = live_game()
    while dealer_should_hit(g.dealer_cards, dealer_hits_soft_17):
        g.dealer_cards.append(draw_from_live_shoe())


def finish_live_round():
    g = live_game()
    hand = g.player_hand

    if hand.surrendered:
        refund = hand.bet / 2
        g.bankroll += refund
        hand.result_text = "Resa"
        hand.payout_units = -0.5
    else:
        if not is_bust(hand.cards):
            dealer_play_live(st.session_state.dealer_hits_soft_17)

        dealer_bj = is_blackjack(g.dealer_cards)
        player_bj = is_blackjack(hand.cards)
        dealer_total, _ = hand_total(g.dealer_cards)
        player_total, _ = hand_total(hand.cards)

        if is_bust(hand.cards):
            hand.result_text = "Sballato"
            hand.payout_units = -1.0 if not hand.doubled else -2.0
        elif player_bj and not dealer_bj:
            payout = hand.bet * (1.0 + BLACKJACK_PAYOUT)
            g.bankroll += payout
            hand.result_text = "Blackjack"
            hand.payout_units = BLACKJACK_PAYOUT
        elif dealer_bj and not player_bj:
            hand.result_text = "Perde contro blackjack del banco"
            hand.payout_units = -1.0 if not hand.doubled else -2.0
        elif dealer_total > 21:
            payout = hand.bet * 2
            g.bankroll += payout
            hand.result_text = "Vince (banco sballa)"
            hand.payout_units = 1.0 if not hand.doubled else 2.0
        elif player_total > dealer_total:
            payout = hand.bet * 2
            g.bankroll += payout
            hand.result_text = "Vince"
            hand.payout_units = 1.0 if not hand.doubled else 2.0
        elif player_total < dealer_total:
            hand.result_text = "Perde"
            hand.payout_units = -1.0 if not hand.doubled else -2.0
        else:
            g.bankroll += hand.bet
            hand.result_text = "Push"
            hand.payout_units = 0.0

    hand.finished = True
    g.round_over = True
    g.in_round = False


def live_action_hit_once_and_finish():
    g = live_game()
    if g.round_over or not g.player_hand.cards or g.player_hand.one_hit_used:
        return

    g.player_hand.cards.append(draw_from_live_shoe())
    g.player_hand.one_hit_used = True
    finish_live_round()


def live_action_stand():
    finish_live_round()


def live_action_surrender():
    g = live_game()
    hand = g.player_hand
    if len(hand.cards) != 2 or g.round_over:
        return
    hand.surrendered = True
    finish_live_round()


def reset_live_round_only():
    g = live_game()
    g.player_hand = LiveHand(cards=[], bet=DEFAULT_BET)
    g.dealer_cards = []
    g.in_round = False
    g.round_over = False
    g.message = ""
    g.frozen_pre_move = None


# ============================================================
# UI HELPERS ANALYTICS
# ============================================================
def build_live_analysis(
    player_cards: List[str],
    dealer_upcard: Optional[str],
    simulations: int,
    n_decks: int,
    dealer_hits_soft_17: bool,
    allow_surrender: bool,
    allow_double: bool,
    allow_split: bool,
    seed: int,
):
    if not player_cards:
        return None

    engine = BlackjackStudy(
        simulations=simulations,
        n_decks=n_decks,
        seed=seed,
        dealer_hits_soft_17=dealer_hits_soft_17,
        allow_surrender=allow_surrender and len(player_cards) == 2,
        allow_double=allow_double and len(player_cards) == 2,
        allow_split=allow_split and len(player_cards) == 2,
        dealer_upcard=dealer_upcard,
    )

    recommended, all_results = engine.recommended_move_by_win_prob(player_cards)
    executed = engine.simulate_recommended_move(player_cards, recommended.move)
    best_ev = engine.best_move_by_ev(all_results)

    third_df = None
    weighted = None
    if recommended.move == "CARTA" and len(player_cards) == 2:
        outcomes = engine.third_card_breakdown(player_cards)
        rows = []
        for item in outcomes:
            rows.append(
                {
                    "Terza carta": item.third_card,
                    "Prob. estrazione": round(item.probability * 100, 2),
                    "Nuova mano": item.resulting_hand,
                    "Totale": item.total,
                    "Soft": item.soft,
                    "Bust": item.bust,
                    "Prob. vittoria dopo terza carta %": round(item.win_prob_after_third * 100, 2),
                    "Prob. pareggio dopo terza carta %": round(item.push_prob_after_third * 100, 2),
                    "Prob. sconfitta dopo terza carta %": round(item.loss_prob_after_third * 100, 2),
                    "EV dopo terza carta": round(item.ev_after_third, 3),
                }
            )
        third_df = pd.DataFrame(rows)
        weights = third_df["Prob. estrazione"] / 100.0
        weighted = {
            "win": float((weights * (third_df["Prob. vittoria dopo terza carta %"] / 100.0)).sum() * 100.0),
            "push": float((weights * (third_df["Prob. pareggio dopo terza carta %"] / 100.0)).sum() * 100.0),
            "loss": float((weights * (third_df["Prob. sconfitta dopo terza carta %"] / 100.0)).sum() * 100.0),
            "ev": float((weights * third_df["EV dopo terza carta"]).sum()),
        }

    return {
        "recommended": recommended,
        "executed": executed,
        "best_ev": best_ev,
        "all_results": all_results,
        "details": format_details(all_results),
        "third_df": third_df,
        "weighted": weighted,
    }


# ============================================================
# APP
# ============================================================
if "app_mode" not in st.session_state:
    st.session_state.app_mode = "Gioca attivamente"

ensure_live_state()

with st.sidebar:
    st.header("Navigazione")
    selected_mode = st.radio(
        "Seleziona modalità",
        ["Gioca attivamente", "Vedi tabelle simulazioni"],
        index=0 if st.session_state.app_mode == "Gioca attivamente" else 1,
    )
    st.session_state.app_mode = selected_mode

    st.header("Parametri Monte Carlo")
    simulations = st.number_input(
        "Numero di simulazioni",
        min_value=100,
        max_value=200000,
        value=DEFAULT_SIMULATIONS,
        step=100,
        key="global_simulations",
    )
    n_decks = st.selectbox("Numero di mazzi", [1, 2, 4, 6, 8], index=3, key="global_decks")
    dealer_hits_soft_17 = st.toggle("Dealer chiede su soft 17", value=False, key="dealer_hits_soft_17")
    allow_double = st.toggle("Abilita raddoppio", value=True, key="allow_double")
    allow_surrender = st.toggle("Abilita resa", value=True, key="allow_surrender")
    allow_split = st.toggle("Abilita split", value=True, key="allow_split")
    seed = st.number_input("Seed casuale simulazioni", min_value=0, max_value=999999, value=DEFAULT_SEED, step=1, key="global_seed")

top1, top2 = st.columns(2)
with top1:
    if st.button("🎮 Vai a Gioca attivamente", use_container_width=True):
        st.session_state.app_mode = "Gioca attivamente"
        st.rerun()
with top2:
    if st.button("📊 Vai a Tabelle simulazioni", use_container_width=True):
        st.session_state.app_mode = "Vedi tabelle simulazioni"
        st.rerun()

# ============================================================
# MODALITÀ STUDIO
# ============================================================
if st.session_state.app_mode == "Vedi tabelle simulazioni":
    st.title("📊 Blackjack: studio Monte Carlo")

    study_upcard_choice = st.selectbox(
        "Carta scoperta del banco",
        UPCARD_OPTIONS,
        index=0,
        key="study_dealer_upcard_choice",
    )

    run_study_btn = st.button("Calcola studio", type="primary", use_container_width=False)

    if run_study_btn:
        with st.spinner("Simulazione in corso..."):
            study_df = run_study(
                simulations=int(simulations),
                n_decks=int(n_decks),
                dealer_hits_soft_17=dealer_hits_soft_17,
                allow_surrender=allow_surrender,
                allow_double=allow_double,
                allow_split=allow_split,
                seed=int(seed),
                dealer_upcard_choice=study_upcard_choice,
            )
        st.session_state["blackjack_compare_df"] = study_df
        st.session_state["blackjack_compare_upcard"] = study_upcard_choice

    if "blackjack_compare_df" in st.session_state:
        df = st.session_state["blackjack_compare_df"].copy()
        current_upcard = st.session_state.get("blackjack_compare_upcard", "Ignora")

        st.subheader("Riepilogo")
        m1, m2, m3 = st.columns(3)
        m1.metric("Coppie iniziali analizzate", len(df))
        m2.metric("Simulazioni per mano", int(simulations))
        m3.metric("Carta banco", format_dealer_upcard_label(current_upcard))

        st.subheader("Filtro")
        hand_options = ["Tutte"] + df["Mano"].tolist()
        selected_hand = st.selectbox("Seleziona una mano iniziale", hand_options)

        if selected_hand != "Tutte":
            df = df[df["Mano"] == selected_hand]

        show_details = st.checkbox("Mostra dettagli completi", value=False, key="study_show_details")
        show_ev_columns = st.checkbox("Mostra colonne EV", value=True, key="study_show_ev")

        display_cols = [
            "Carta banco scoperta",
            "Carta 1",
            "Carta 2",
            "Mano",
            "Totale iniziale",
            "Soft",
            "Split consentito",
            "Mossa consigliata",
            "Prob. vittoria prima della mossa %",
            "Prob. pareggio prima della mossa %",
            "Prob. sconfitta prima della mossa %",
        ]

        if show_ev_columns:
            display_cols.extend(
                [
                    "EV prima della mossa",
                    "Mossa migliore per EV",
                    "EV migliore assoluto",
                ]
            )

        if show_details:
            display_cols.append("Dettagli tutte le mosse")

        st.subheader("Tabella risultati")
        st.dataframe(df[display_cols].reset_index(drop=True), use_container_width=True, hide_index=True)

        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Scarica CSV principale",
            data=csv_data,
            file_name="blackjack_confronto_con_carta_banco.csv",
            mime="text/csv",
        )

        st.subheader("Dettaglio mano")
        detail_options = st.session_state["blackjack_compare_df"]["Mano"].tolist()
        selected_detail = st.selectbox("Scegli una mano per il dettaglio", detail_options, key="study_detail_hand")

        row = st.session_state["blackjack_compare_df"][
            st.session_state["blackjack_compare_df"]["Mano"] == selected_detail
        ].iloc[0]

        d1, d2, d3 = st.columns(3)
        d1.metric("Carta banco", row["Carta banco scoperta"])
        d2.metric("Mossa consigliata", row["Mossa consigliata"])
        d3.metric("Win% prima", f"{row['Prob. vittoria prima della mossa %']:.2f}%")

        d4, d5 = st.columns(2)
        d4.metric("Push% prima", f"{row['Prob. pareggio prima della mossa %']:.2f}%")
        d5.metric("Loss% prima", f"{row['Prob. sconfitta prima della mossa %']:.2f}%")

        if show_ev_columns:
            e1, e2 = st.columns(2)
            e1.metric("EV prima", f"{row['EV prima della mossa']:.3f}")
            e2.metric("Mossa migliore per EV", row["Mossa migliore per EV"])

        if show_details:
            st.caption(row["Dettagli tutte le mosse"])

        if row["Mossa consigliata"] == "CARTA":
            st.subheader("Dettaglio: probabilità dopo la terza carta")

            third_df = run_third_card_breakdown(
                simulations=int(simulations),
                n_decks=int(n_decks),
                dealer_hits_soft_17=dealer_hits_soft_17,
                allow_surrender=allow_surrender,
                allow_double=allow_double,
                allow_split=allow_split,
                seed=int(seed),
                card1=row["Carta 1"],
                card2=row["Carta 2"],
                dealer_upcard_choice=current_upcard,
            )

            st.dataframe(third_df, use_container_width=True, hide_index=True)

            csv_third = third_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Scarica CSV terza carta",
                data=csv_third,
                file_name=f"blackjack_terza_carta_{row['Carta 1']}_{row['Carta 2']}_bank_{current_upcard}.csv",
                mime="text/csv",
            )

            weights = third_df["Prob. estrazione"] / 100.0
            weighted_win = (weights * (third_df["Prob. vittoria dopo terza carta %"] / 100.0)).sum() * 100.0
            weighted_push = (weights * (third_df["Prob. pareggio dopo terza carta %"] / 100.0)).sum() * 100.0
            weighted_loss = (weights * (third_df["Prob. sconfitta dopo terza carta %"] / 100.0)).sum() * 100.0
            weighted_ev = (weights * third_df["EV dopo terza carta"]).sum()

            win_before = float(row["Prob. vittoria prima della mossa %"])
            weighted_diff = weighted_win - win_before

            w1, w2 = st.columns(2)
            w1.metric("Win% pesata dopo estrazione della terza carta", f"{weighted_win:.2f}%")
            w2.metric("Differenza vs Win% prima", f"{weighted_diff:.2f}%")

            w3, w4, w5 = st.columns(3)
            w3.metric("Push% pesata dopo estrazione della terza carta", f"{weighted_push:.2f}%")
            w4.metric("Loss% pesata dopo estrazione della terza carta", f"{weighted_loss:.2f}%")
            w5.metric("EV pesato dopo estrazione della terza carta", f"{weighted_ev:.3f}")
        else:
            st.info("Per questa mano la mossa consigliata non è CARTA.")
    else:
        st.info("Premi **Calcola studio** per generare le tabelle.")

# ============================================================
# MODALITÀ GIOCO ATTIVO
# ============================================================
else:
    st.title("🎮 Blackjack: gioco attivo con suggerimenti Monte Carlo")
    st.caption(
        "Nel gioco attivo puoi pescare una sola carta. Dopo l’estrazione, la mano si chiude "
        "automaticamente con STAI e viene mostrato il risultato finale. Le probabilità tengono "
        "conto anche della carta scoperta del banco."
    )

    g = live_game()
    if g.n_decks != int(n_decks):
        reset_live_shoe(int(n_decks))

    left, right = st.columns([1.6, 1])

    with left:
        st.subheader("Tavolo di gioco")
        b1, b2, b3 = st.columns(3)
        with b1:
            st.metric("Bankroll", f"€ {g.bankroll:,.2f}")
        with b2:
            st.metric("Carte nel sabot", len(g.shoe))
        with b3:
            st.metric("Mazzi", g.n_decks)

        max_bet = float(max(1, int(g.bankroll))) if g.bankroll >= 1 else 1.0
        default_bet = float(min(max(1, int(DEFAULT_BET)), max(1, int(g.bankroll)))) if g.bankroll >= 1 else 1.0

        bet = st.number_input(
            "Puntata",
            min_value=1.0,
            max_value=max_bet,
            value=default_bet,
            step=1.0,
            key="live_bet",
        )

        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("Distribuisci", use_container_width=True, disabled=g.bankroll < 1 or (g.in_round and not g.round_over)):
                start_live_round(
                    bet=float(bet),
                    simulations=int(simulations),
                    n_decks=int(n_decks),
                    dealer_hits_soft_17=dealer_hits_soft_17,
                    allow_surrender=allow_surrender,
                    allow_double=allow_double,
                    allow_split=allow_split,
                    seed=int(seed),
                )
                st.rerun()
        with c2:
            if st.button("Nuovo sabot", use_container_width=True):
                reset_live_shoe(int(n_decks))
                st.rerun()
        with c3:
            if st.button("Reset bankroll", use_container_width=True):
                live_seed = generate_live_seed()
                st.session_state.live_game = LiveGameState(
                    bankroll=INITIAL_BANKROLL,
                    shoe=fresh_shoe_list(int(n_decks), random.Random(live_seed)),
                    dealer_cards=[],
                    player_hand=LiveHand(cards=[], bet=DEFAULT_BET),
                    in_round=False,
                    round_over=False,
                    message="",
                    n_decks=int(n_decks),
                    seed=live_seed,
                    frozen_pre_move=None,
                )
                st.rerun()

        table_img = build_blackjack_table_image(
            dealer_cards=g.dealer_cards,
            player_cards=g.player_hand.cards,
            round_over=g.round_over,
            player_bet=g.player_hand.bet,
            bankroll=g.bankroll,
            result_text=g.player_hand.result_text,
            payout_units=g.player_hand.payout_units,
        )
        st.image(table_img, use_container_width=True)

        if g.message:
            st.caption(g.message)

        if g.player_hand.cards and not g.round_over:
            st.markdown("### Azioni")
            a1, a2, a3 = st.columns(3)
            with a1:
                if st.button(
                    "Carta (una sola)",
                    use_container_width=True,
                    disabled=g.player_hand.one_hit_used
                ):
                    live_action_hit_once_and_finish()
                    st.rerun()
            with a2:
                if st.button("Stai", use_container_width=True):
                    live_action_stand()
                    st.rerun()
            with a3:
                if st.button("Resa", use_container_width=True, disabled=(len(g.player_hand.cards) != 2 or not allow_surrender)):
                    live_action_surrender()
                    st.rerun()

        if g.round_over:
            if st.button("Nuova mano", type="primary", use_container_width=True):
                reset_live_round_only()
                st.rerun()

    with right:
        st.subheader("Suggerimento in tempo reale")
        if g.player_hand.cards:
            dealer_upcard = g.dealer_cards[0] if g.dealer_cards else None

            analysis = build_live_analysis(
                player_cards=g.player_hand.cards,
                dealer_upcard=dealer_upcard,
                simulations=int(simulations),
                n_decks=int(n_decks),
                dealer_hits_soft_17=dealer_hits_soft_17,
                allow_surrender=allow_surrender,
                allow_double=allow_double,
                allow_split=allow_split,
                seed=int(seed),
            )

            frozen = g.frozen_pre_move
            if frozen is None:
                frozen = {
                    "move": analysis["recommended"].move,
                    "win": analysis["recommended"].win_prob,
                    "push": analysis["recommended"].push_prob,
                    "loss": analysis["recommended"].loss_prob,
                    "ev": analysis["recommended"].ev,
                    "best_ev_move": analysis["best_ev"].move,
                    "details": analysis["details"],
                    "dealer_upcard": dealer_upcard,
                }

            exe = analysis["executed"]

            if dealer_upcard is not None:
                st.metric("Carta scoperta del banco", dealer_upcard)

            r1, r2 = st.columns(2)
            r1.metric("Mossa migliore", frozen["move"])
            r2.metric("Mossa migliore per EV", frozen["best_ev_move"])

            s1, s2, s3 = st.columns(3)
            s1.metric("Win% prima", f"{frozen['win'] * 100:.2f}%")
            s2.metric("Push% prima", f"{frozen['push'] * 100:.2f}%")
            s3.metric("Loss% prima", f"{frozen['loss'] * 100:.2f}%")

            t1, t2, t3 = st.columns(3)
            t1.metric("Win% dopo", f"{exe.win_prob * 100:.2f}%")
            t2.metric("Push% dopo", f"{exe.push_prob * 100:.2f}%")
            t3.metric("Loss% dopo", f"{exe.loss_prob * 100:.2f}%")

            u1, u2 = st.columns(2)
            u1.metric("EV prima", f"{frozen['ev']:.3f}")
            u2.metric("EV dopo", f"{exe.ev:.3f}")

            st.caption(frozen["details"])

            if analysis["third_df"] is not None and len(g.player_hand.cards) == 2:
                st.markdown("### Dettaglio prossima carta")
                st.dataframe(analysis["third_df"], use_container_width=True, hide_index=True)

                weighted = analysis["weighted"]
                if weighted is not None:
                    v1, v2 = st.columns(2)
                    v1.metric("Win% pesata dopo terza carta", f"{weighted['win']:.2f}%")
                    v2.metric("Differenza vs Win% prima", f"{weighted['win'] - frozen['win'] * 100:.2f}%")

                    v3, v4, v5 = st.columns(3)
                    v3.metric("Push% pesata", f"{weighted['push']:.2f}%")
                    v4.metric("Loss% pesata", f"{weighted['loss']:.2f}%")
                    v5.metric("EV pesato", f"{weighted['ev']:.3f}")
        else:
            st.info("Distribuisci una mano per vedere i suggerimenti live.")
