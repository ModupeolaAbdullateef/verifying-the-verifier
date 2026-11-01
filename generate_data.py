"""
generate_data.py - Synthetic dataset generator.

Part of the artefact for the study "Verifying the Verifier: Evaluating an
Automated Verification Layer for Reducing Errors and Hallucinations in an
Enterprise Multi-Agent Workflow".

The script produces the following files under ./data:

  reference/suppliers.json          Supplier master: the reference table
                                    against which the normaliser agent
                                    matches supplier names.
  reference/purchase_orders.json    Purchase-order register: the approved
                                    amount for each purchase order, used by
                                    the decision agent.
  documents/DOC-L01.txt ...         21 invoices as plain text (7 low, 7 medium,
                                    7 high complexity): the exact printed content.
  images/DOC-L01_p1.jpg ...         The same invoices as scanned page images, the
                                    input to the workflow. Scan quality degrades
                                    with complexity (see SCAN_PROFILES).
  reference/services.json           IT service catalogue: the reference table
                                    against which the routing agent matches
                                    the affected service of a ticket.
  tickets/TKT-L01.txt ...           10 IT service tickets for the replication
                                    study (3 low, 4 medium, 3 high complexity).
  ground_truth/documents.json       Correct values for every invoice, both at
                                    the verification gate and after
                                    normalisation, plus the correct decision.
  ground_truth/tickets.json         Correct labels for every ticket.
  manifest.json                     Seed, counts, human-review subsets,
                                    business rules and field conventions.

All values derive from a fixed random seed (SEED), so repeated runs produce
byte-for-byte identical output. Changing SEED produces a different dataset.

All organisations, people and values are fictional.

Usage:
    python generate_data.py
"""

from __future__ import annotations

import json
import os
import random
import textwrap
from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------

SEED = 20261101
GENERATOR_VERSION = "3.1.0"

OUT = Path("data")
VAT_RATE = Decimal("0.20")
APPROVAL_LIMIT = Decimal("10000.00")  # totals above this need manager sign-off
PO_TOLERANCE = Decimal("0.50")        # allowed difference between invoice and PO

BUYER_NAME = "Arden Valley Communications plc"
BUYER_ADDRESS = "Accounts Payable, 1 Riverside Way, Leeds LS1 4AP"
CUSTOMER_REF_PREFIX = "CC"            # buyer's internal cost-centre reference
BUYER_VAT = "GB 914 2275 08"          # buyer's own VAT number (printed on some invoices)

# Simulated scan quality by complexity level: font size (px), skew (degrees),
# blur radius, JPEG quality, share of speckled pixels, ink grey level (0 = black)
# and paper grey level (255 = white).
SCAN_PROFILES = {
    "low":    {"font_px": 24, "skew": 0.0, "blur": 0.0, "jpeg": 90, "speckle": 0.000, "ink": 20, "paper": 255},
    "medium": {"font_px": 21, "skew": 0.5, "blur": 0.6, "jpeg": 50, "speckle": 0.004, "ink": 55, "paper": 246},
    "high":   {"font_px": 19, "skew": 1.1, "blur": 0.9, "jpeg": 35, "speckle": 0.010, "ink": 85, "paper": 236},
}
SCAN_WRAP_CHARS = 96        # characters per printed line on the page image

# Line items per invoice, by complexity level.
ITEM_COUNTS = {"low": (2, 4), "medium": (8, 14), "high": (15, 24)}

# Qualifiers that make repeated catalogue items distinct (site, period or phase),
# as on real invoices for recurring services delivered to several locations.
QUALIFIERS = ["Leeds HQ", "Bradford depot", "York office", "Wakefield site", "Floor 2",
              "Floor 4", "Server room", "Reception", "January", "February", "March",
              "April", "May", "June", "Phase 1", "Phase 2", "Batch A", "Batch B"]

# Human-in-the-loop subsets, stratified by complexity (3 low, 4 medium, 3 high
# documents; 2 low, 2 medium, 1 high tickets).
HITL_DOCUMENTS = ["DOC-L01", "DOC-L02", "DOC-L05",
                  "DOC-M01", "DOC-M03", "DOC-M04", "DOC-M06",
                  "DOC-H02", "DOC-H04", "DOC-H06"]
HITL_TICKETS = ["TKT-L01", "TKT-L03", "TKT-M02", "TKT-M04", "TKT-H01"]

TWO_PLACES = Decimal("0.01")


def money(value) -> Decimal:
    """Round to pence using conventional (half-up) rounding."""
    return Decimal(value).quantize(TWO_PLACES, rounding=ROUND_HALF_UP)


# --------------------------------------------------------------------------
# Reference data
# --------------------------------------------------------------------------

# (supplier_id, legal name, currency, VAT country prefix or None, email domain)
# Pairs such as SUP-001/SUP-002 are deliberately similar so that supplier
# matching can go wrong in the high-complexity documents.
SUPPLIERS = [
    ("SUP-001", "Northgate Office Supplies Ltd", "GBP", "GB", "northgate-supplies.example"),
    ("SUP-002", "Northgate Office Solutions Ltd", "GBP", "GB", "northgate-solutions.example"),
    ("SUP-003", "Pennine Facilities Management Ltd", "GBP", "GB", "pennine-fm.example"),
    ("SUP-004", "Pennine Facility Services Ltd", "GBP", "GB", "pennine-services.example"),
    ("SUP-005", "Brightwater IT Services Ltd", "GBP", "GB", "brightwater-it.example"),
    ("SUP-006", "Brightwater IT Solutions Ltd", "GBP", "GB", "brightwater-solutions.example"),
    ("SUP-007", "Calder Valley Catering Ltd", "GBP", "GB", "caldervalley-catering.example"),
    ("SUP-008", "Ashdown Print & Design Ltd", "GBP", "GB", "ashdown-print.example"),
    ("SUP-009", "Meridian Logistics UK Ltd", "GBP", "GB", "meridian-logistics.example"),
    ("SUP-010", "Severn Electrical Contractors Ltd", "GBP", "GB", "severn-electrical.example"),
    ("SUP-011", "Highfield Cleaning Co Ltd", "GBP", "GB", "highfield-cleaning.example"),
    ("SUP-012", "Orbis Telecom Hardware Ltd", "GBP", "GB", "orbis-telecom.example"),
    ("SUP-013", "Kestrel Security Systems Ltd", "GBP", "GB", "kestrel-security.example"),
    ("SUP-014", "Lindqvist Software AB", "EUR", "SE", "lindqvist-software.example"),
]

# Catalogue of goods and services per supplier: (description, min price, max price)
CATALOGUE = {
    "SUP-001": [("A4 copier paper, 80gsm (box of 5 reams)", 18, 26),
                ("Black toner cartridge, high yield", 45, 95),
                ("Lever-arch files (pack of 10)", 12, 22),
                ("Desk organiser set", 15, 35),
                ("Whiteboard markers (pack of 12)", 6, 14),
                ("Ergonomic office chair", 140, 320),
                ("Monitor riser", 25, 55),
                ("Shredder, cross-cut", 90, 220),
                ("Sticky notes (pack of 24)", 8, 16)],
    "SUP-002": [("Hot-desk booking licence (annual)", 900, 2400),
                ("Meeting-room display installation", 1200, 3200),
                ("Office layout consultancy (per day)", 450, 750)],
    "SUP-003": [("Planned maintenance visit", 220, 480),
                ("HVAC filter replacement", 95, 210),
                ("Emergency call-out (out of hours)", 260, 540),
                ("Fire-door inspection", 150, 300),
                ("Lift service contract (monthly)", 600, 1100),
                ("Water hygiene testing", 140, 290),
                ("Roof gutter clearance", 180, 420),
                ("Boiler annual service", 210, 380),
                ("Car park line marking", 350, 900)],
    "SUP-004": [("Reception cover (per day)", 160, 240),
                ("Post-room services (monthly)", 700, 1300)],
    "SUP-005": [("Managed service desk (monthly)", 2500, 4800),
                ("Laptop imaging and deployment (per device)", 35, 70),
                ("On-site engineer (per day)", 420, 680),
                ("Microsoft 365 migration workshop", 900, 1800),
                ("Firewall configuration review", 750, 1500),
                ("Docking station", 110, 210),
                ("Laptop battery replacement", 70, 140),
                ("Backup monitoring (monthly)", 300, 650),
                ("Endpoint security licence (annual, per seat)", 25, 60)],
    "SUP-006": [("Network design consultancy (per day)", 650, 950),
                ("Cloud cost review", 1200, 2600)],
    "SUP-007": [("Working lunch for 12 people", 120, 210),
                ("Conference catering (per head)", 18, 35),
                ("Tea and coffee service (weekly)", 60, 120),
                ("Breakfast platter", 45, 90),
                ("Fruit bowl delivery (weekly)", 20, 40),
                ("Afternoon tea for 20 people", 150, 260),
                ("Sandwich platter", 55, 110),
                ("Bottled water (case of 24)", 9, 18),
                ("Event staffing (per hour)", 18, 30)],
    "SUP-008": [("Business cards (box of 500)", 25, 60),
                ("A1 exhibition poster", 35, 80),
                ("Roller banner with stand", 85, 160),
                ("Brochure print run (1,000 copies)", 450, 1100),
                ("Graphic design (per hour)", 45, 85),
                ("Letterheads (box of 1,000)", 60, 140),
                ("Branded notepads (pack of 50)", 70, 150),
                ("Large-format vinyl signage", 120, 340),
                ("Proofing and colour matching", 40, 95)],
    "SUP-009": [("Next-day pallet delivery", 65, 140),
                ("Courier collection, same day", 30, 75),
                ("Warehouse storage (per pallet, monthly)", 12, 25),
                ("International freight, Europe", 380, 900),
                ("Pallet wrapping and labelling", 15, 35)],
    "SUP-010": [("PAT testing (per item)", 2, 5),
                ("Emergency lighting test", 180, 360),
                ("Socket and data-point installation", 85, 160),
                ("Distribution board upgrade", 1400, 3200),
                ("Electrician (per day)", 280, 420),
                ("LED panel replacement", 35, 75),
                ("EV charger installation", 900, 1800),
                ("Fixed wire testing (per circuit)", 8, 18),
                ("Electrical safety certificate", 150, 320)],
    "SUP-011": [("Office cleaning (weekly)", 220, 460),
                ("Deep clean, kitchen areas", 160, 320),
                ("Window cleaning, external", 180, 420),
                ("Washroom consumables restock", 60, 140),
                ("Carpet shampoo (per floor)", 220, 480)],
    "SUP-012": [("IP desk phone", 85, 160),
                ("Headset, noise-cancelling", 60, 140),
                ("24-port PoE switch", 380, 820),
                ("Fibre patch leads (pack of 10)", 40, 90),
                ("Wireless access point", 180, 360),
                ("Rack-mount UPS", 650, 1400),
                ("SFP+ transceiver", 30, 90),
                ("Cat6 cable (305 m box)", 95, 180),
                ("Conference speakerphone", 180, 420)],
    "SUP-013": [("CCTV camera, dome", 140, 320),
                ("Access-control card readers", 220, 480),
                ("Alarm system service", 180, 380),
                ("Security patrol (per night)", 160, 260),
                ("Access cards (pack of 50)", 75, 140),
                ("Intruder alarm keypad", 120, 260),
                ("Video door entry unit", 350, 780),
                ("Key-holding service (monthly)", 60, 140),
                ("Security risk assessment", 450, 950)],
    "SUP-014": [("Workflow platform subscription (annual)", 4200, 7800),
                ("Implementation consultancy (per day)", 900, 1300),
                ("Premium support add-on (annual)", 1100, 2200),
                ("Additional user licences (pack of 10)", 600, 1200),
                ("Data migration service", 1500, 3200),
                ("Training session (half day)", 450, 800),
                ("API integration module (annual)", 900, 1900),
                ("Sandbox environment (annual)", 700, 1400),
                ("Single sign-on connector (annual)", 500, 1100)],
}

FIRST_NAMES = ["Amara", "Ben", "Chloe", "Daniel", "Efe", "Fatima", "George",
               "Hannah", "Imran", "Jess", "Kwame", "Laura", "Mohammed",
               "Niamh", "Oliver", "Priya", "Rhys", "Sade", "Tom", "Zainab"]
LAST_NAMES = ["Adeyemi", "Bennett", "Clarke", "Davies", "Evans", "Fraser",
              "Green", "Hughes", "Iqbal", "Jones", "Khan", "Lewis", "Morgan",
              "Nwosu", "Owen", "Patel", "Roberts", "Shah", "Taylor", "Walsh"]
STREETS = ["Mill Lane", "Station Road", "Canal Street", "Park View", "Bridge Street",
           "Wharf Road", "Victoria Terrace", "Kingsway", "Market Place", "Albion Road"]
TOWNS = [("Bradford", "BD1"), ("Huddersfield", "HD1"), ("Halifax", "HX1"),
         ("Wakefield", "WF1"), ("York", "YO1"), ("Sheffield", "S1"),
         ("Manchester", "M1"), ("Harrogate", "HG1"), ("Stockholm", None)]

MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]

# --------------------------------------------------------------------------
# Data structures
# --------------------------------------------------------------------------


@dataclass
class LineItem:
    description: str
    quantity: int
    unit_price: Decimal

    @property
    def line_total(self) -> Decimal:
        return money(self.unit_price * self.quantity)


@dataclass
class Invoice:
    doc_id: str
    complexity: str              # "low" | "medium" | "high"
    layout: str                  # "A" classic | "B" key-value | "C" email
    supplier_id: str
    supplier_name: str           # canonical name in the supplier master
    printed_supplier_name: str   # name exactly as it appears on the invoice
    supplier_vat: str | None
    supplier_email: str
    supplier_address: str
    currency: str
    invoice_number: str
    invoice_date: date
    payment_terms_days: int
    po_number: str
    customer_reference: str | None
    items: list[LineItem]
    reverse_charge: bool = False
    original_total: Decimal | None = None   # superseded total (high only)
    order_date: date | None = None          # distractor date (high only)
    delivery_date: date | None = None       # distractor date (high only)
    credit_note: str | None = None
    po_amount: Decimal = Decimal("0")
    supplier_ref: str | None = None          # supplier's own reference (distractor)
    delivery_note: str | None = None         # delivery-note number (distractor)
    show_buyer_vat: bool = False             # buyer's VAT number printed (distractor)
    balance_bf: Decimal | None = None        # balance brought forward from earlier invoices
    page_break_after: int | None = None      # line items on page 1 of a two-page invoice
    previous_invoice: dict | None = None     # earlier, settled invoice mentioned (distractor)
    has_discount: bool = False

    @property
    def subtotal(self) -> Decimal:
        return money(sum((i.line_total for i in self.items), Decimal("0")))

    @property
    def vat_amount(self) -> Decimal:
        return Decimal("0.00") if self.reverse_charge else money(self.subtotal * VAT_RATE)

    @property
    def total(self) -> Decimal:
        return money(self.subtotal + self.vat_amount)

    @property
    def due_date(self) -> date:
        return self.invoice_date + timedelta(days=self.payment_terms_days)


# --------------------------------------------------------------------------
# Formatting helpers
# --------------------------------------------------------------------------

SYMBOL = {"GBP": "£", "EUR": "€"}


def fmt_amount(value: Decimal, currency: str, style: str = "symbol") -> str:
    """Format money in one of several styles seen on real invoices."""
    plain = f"{value:,.2f}"
    if style == "symbol":
        return f"{SYMBOL[currency]}{plain}"
    if style == "code":
        return f"{currency} {plain}"
    if style == "suffix":                      # e.g. 1 284,50 EUR (continental)
        whole, pence = f"{value:.2f}".split(".")
        grouped = f"{int(whole):,}".replace(",", " ")
        return f"{grouped},{pence} {currency}"
    if style == "bare":
        return f"{value:.2f}"
    raise ValueError(style)


def ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def fmt_date(d: date, style: str) -> str:
    """Format dates in the styles used to vary difficulty."""
    if style == "long":        # 12 March 2026 (unambiguous)
        return f"{d.day} {MONTHS[d.month - 1]} {d.year}"
    if style == "ordinal":     # 12th March 2026
        return f"{ordinal(d.day)} {MONTHS[d.month - 1]} {d.year}"
    if style == "uk":          # 12/03/2026 (UK order, ambiguous to some readers)
        return d.strftime("%d/%m/%Y")
    if style == "iso":         # 2026-03-12
        return d.isoformat()
    if style == "short":       # 12-Mar-26
        return f"{d.day:02d}-{MONTHS[d.month - 1][:3]}-{d.strftime('%y')}"
    raise ValueError(style)


# --------------------------------------------------------------------------
# Generator
# --------------------------------------------------------------------------


class Generator:
    def __init__(self, seed: int):
        self.rng = random.Random(seed)
        self.used_invoice_numbers: set[str] = set()
        self.used_po_numbers: set[str] = set()
        self.suppliers = self._build_suppliers()

    # -- reference data ---------------------------------------------------

    def _vat_number(self, prefix: str) -> str:
        digits = "".join(str(self.rng.randint(0, 9)) for _ in range(9))
        if prefix == "GB":
            return f"GB {digits[:3]} {digits[3:7]} {digits[7:]}"
        return f"{prefix}{digits}01"           # Swedish-style VAT number

    def _address(self, foreign: bool) -> str:
        number = self.rng.randint(2, 180)
        street = self.rng.choice(STREETS)
        if foreign:
            return f"Sveavägen {number}, 111 34 Stockholm, Sweden"
        town, district = self.rng.choice([t for t in TOWNS if t[1]])
        inward = f"{self.rng.randint(1, 9)}{self.rng.choice('ABDEFGHJLNPQRSTUWXYZ')}{self.rng.choice('ABDEFGHJLNPQRSTUWXYZ')}"
        return f"Unit {number}, {street}, {town} {district} {inward}"

    def _build_suppliers(self) -> dict[str, dict]:
        table = {}
        for sid, name, currency, vat_prefix, domain in SUPPLIERS:
            table[sid] = {
                "supplier_id": sid,
                "name": name,
                "currency": currency,
                "vat_number": self._vat_number(vat_prefix) if vat_prefix else None,
                "address": self._address(foreign=(vat_prefix == "SE")),
                "email": f"accounts@{domain}",
            }
        return table

    # -- identifiers --------------------------------------------------------

    def _invoice_number(self, supplier_id: str) -> str:
        while True:
            style = self.rng.choice(["INV-{n}", "{n}", "SI/{y}/{n}", "{p}{n}"])
            n = self.rng.randint(10000, 99999)
            prefix = supplier_id.split("-")[1]
            number = style.format(n=n, y=2026, p=f"S{prefix}-")
            if number not in self.used_invoice_numbers:
                self.used_invoice_numbers.add(number)
                return number

    def _po_number(self) -> str:
        while True:
            number = f"PO-{self.rng.randint(400000, 499999)}"
            if number not in self.used_po_numbers:
                self.used_po_numbers.add(number)
                return number

    def _customer_reference(self) -> str:
        return f"{CUSTOMER_REF_PREFIX}-{self.rng.randint(1000, 9999)}"

    # -- name variants (what is printed vs what is in the master) -----------

    def _printed_name(self, name: str, complexity: str) -> str:
        if complexity == "low":
            return name
        variants = [
            name.replace(" Ltd", ""),
            name.replace(" Ltd", " Limited"),
            name.upper(),
            name.replace("&", "and").replace(" Ltd", ""),
            name.replace(" Ltd", " Ltd."),
        ]
        variants = [v for v in variants if v != name] or [name]
        return self.rng.choice(variants)

    # -- line items -----------------------------------------------------------

    def _items(self, supplier_id: str, complexity: str, want_over_limit: bool,
               with_discount: bool) -> list[LineItem]:
        """
        Draw distinct line items until the gross total lands on the required side
        of the approval limit. Medium and high invoices qualify each description
        with a site, period or phase, so catalogue items can recur as distinct lines.
        """
        catalogue = CATALOGUE[supplier_id]
        low, high = ITEM_COUNTS[complexity]
        for _ in range(2000):
            count = self.rng.randint(low, high)
            if complexity == "low":
                picks = [(d, lo, hi) for d, lo, hi in
                         self.rng.sample(catalogue, k=min(count, len(catalogue)))]
            else:
                combos = [(d, q) for d, _, _ in catalogue for q in QUALIFIERS]
                chosen = self.rng.sample(combos, k=count)
                bounds = {d: (lo, hi) for d, lo, hi in catalogue}
                picks = [(f"{d} - {q}", *bounds[d]) for d, q in chosen]
            items = []
            for desc, lo, hi in picks:
                price = money(Decimal(self.rng.randint(lo * 100, hi * 100)) / 100)
                qty = self.rng.choice([1, 1, 1, 2, 3, 4, 5, 6, 10, 12])
                if price > 400:
                    qty = self.rng.choice([1, 1, 2])
                items.append(LineItem(desc, qty, price))

            def gross(lines):
                net = sum(i.line_total for i in lines)
                if with_discount:
                    net -= money(net * Decimal("0.05"))
                return money(net * (1 + VAT_RATE))

            if want_over_limit:
                top = max(items, key=lambda i: i.unit_price)
                while gross(items) <= APPROVAL_LIMIT * Decimal("1.05") and top.quantity < 40:
                    top.quantity += 1
            else:
                # Reduce quantities of the most expensive lines until under the limit.
                for item in sorted(items, key=lambda i: i.unit_price, reverse=True):
                    while gross(items) >= APPROVAL_LIMIT * Decimal("0.90") and item.quantity > 1:
                        item.quantity -= 1
            total = gross(items)
            if (want_over_limit and total > APPROVAL_LIMIT * Decimal("1.05")) or \
               (not want_over_limit and total < APPROVAL_LIMIT * Decimal("0.90")):
                if with_discount:
                    net = sum(i.line_total for i in items)
                    items.append(LineItem("Less: loyalty discount 5%", 1,
                                          -money(net * Decimal("0.05"))))
                return items
        raise RuntimeError(f"Could not build items for {supplier_id}")

    # -- invoices ---------------------------------------------------------------

    def invoice(self, doc_id: str, complexity: str, supplier_id: str,
                layout: str, over_limit: bool) -> Invoice:
        s = self.suppliers[supplier_id]
        with_discount = complexity != "low" and self.rng.random() < 0.5
        items = self._items(supplier_id, complexity, over_limit, with_discount)

        inv_date = date(2026, 1, 5) + timedelta(days=self.rng.randint(0, 250))
        terms = self.rng.choice([14, 30, 30, 30, 45, 60])

        invoice = Invoice(
            doc_id=doc_id,
            complexity=complexity,
            layout=layout,
            supplier_id=supplier_id,
            supplier_name=s["name"],
            printed_supplier_name=self._printed_name(s["name"], complexity),
            supplier_vat=s["vat_number"],
            supplier_email=s["email"],
            supplier_address=s["address"],
            currency=s["currency"],
            invoice_number=self._invoice_number(supplier_id),
            invoice_date=inv_date,
            payment_terms_days=terms,
            po_number=self._po_number(),
            customer_reference=None if complexity == "high" else self._customer_reference(),
            items=items,
            reverse_charge=(s["currency"] != "GBP"),
            has_discount=with_discount,
        )

        # Distractor references printed near the true ones.
        invoice.supplier_ref = f"SR-{self.rng.randint(10000, 99999)}"
        if complexity != "low":
            invoice.delivery_note = f"DN-{self.rng.randint(100000, 999999)}"
            invoice.show_buyer_vat = True
            if self.rng.random() < 0.5:
                invoice.balance_bf = money(Decimal(self.rng.randint(25000, 900000)) / 100)

        if complexity == "high":
            # Distractor dates close to the invoice date.
            invoice.order_date = inv_date - timedelta(days=self.rng.randint(2, 9))
            invoice.delivery_date = inv_date - timedelta(days=self.rng.randint(0, 2))
            # The PO approved an earlier, higher amount; a credit note reduced it.
            uplift = money(Decimal(self.rng.randint(12000, 90000)) / 100)
            invoice.original_total = money(invoice.total + uplift)
            invoice.credit_note = f"CN-{self.rng.randint(1000, 9999)}"
            invoice.po_amount = invoice.original_total
            # A two-page layout with a carried-forward subtotal.
            invoice.page_break_after = len(items) // 2 if layout in ("A", "B") else None
            # An earlier, already settled invoice from the same supplier.
            prev_date = inv_date - timedelta(days=self.rng.randint(25, 40))
            invoice.previous_invoice = {
                "number": self._invoice_number(supplier_id),
                "date": prev_date,
                "po_number": self._po_number(),
                "total": money(Decimal(self.rng.randint(30000, 900000)) / 100),
            }
        else:
            invoice.po_amount = invoice.total
        return invoice


# --------------------------------------------------------------------------
# Business rules (also written into the manifest for the decision agent)
# --------------------------------------------------------------------------

BUSINESS_RULES = [
    {"order": 1, "code": "HOLD_PO_NOT_FOUND",
     "rule": "Hold if the purchase-order number is not in the purchase-order register."},
    {"order": 2, "code": "HOLD_SUPPLIER_MISMATCH",
     "rule": "Hold if the matched supplier_id differs from the supplier on the purchase order."},
    {"order": 3, "code": "HOLD_PO_MISMATCH",
     "rule": f"Hold if the invoice total differs from the purchase-order approved amount by more than {PO_TOLERANCE}."},
    {"order": 4, "code": "HOLD_APPROVAL_LIMIT",
     "rule": f"Hold for manager approval if the invoice total exceeds {APPROVAL_LIMIT} in the invoice currency."},
    {"order": 5, "code": "APPROVE",
     "rule": "Otherwise approve for payment."},
]


def decide(invoice: Invoice) -> str:
    """Apply the business rules in order to the ground-truth invoice."""
    if abs(invoice.total - invoice.po_amount) > PO_TOLERANCE:
        return "HOLD_PO_MISMATCH"
    if invoice.total > APPROVAL_LIMIT:
        return "HOLD_APPROVAL_LIMIT"
    return "APPROVE"


# --------------------------------------------------------------------------
# Rendering: three layouts, with difficulty controlled by complexity
# --------------------------------------------------------------------------


def date_style(inv: Invoice, rng: random.Random) -> str:
    return {"low": "long", "medium": rng.choice(["uk", "short"]),
            "high": rng.choice(["ordinal", "uk"])}[inv.complexity]


def amount_style(inv: Invoice, rng: random.Random) -> str:
    if inv.complexity == "low":
        return "symbol"
    if inv.currency == "EUR":
        return rng.choice(["suffix", "code"])
    return rng.choice(["symbol", "code"])


def terms_line(inv: Invoice, rng: random.Random, dstyle: str) -> str:
    """Low and high state the due date; medium gives terms only (due date must be derived)."""
    if inv.complexity == "medium":
        return rng.choice([f"Payment terms: {inv.payment_terms_days} days from date of invoice.",
                           f"Terms: net {inv.payment_terms_days} days.",
                           f"Please pay within {inv.payment_terms_days} days of the invoice date."])
    if inv.complexity == "high":
        return (f"Terms: {inv.payment_terms_days} days. Payment due by "
                f"{fmt_date(inv.due_date, 'iso')}.")
    return (f"Payment terms: {inv.payment_terms_days} days. "
            f"Due date: {fmt_date(inv.due_date, dstyle)}.")


def high_notes(inv: Invoice, astyle: str) -> list[str]:
    """Distractor lines for high-complexity invoices."""
    prev = inv.previous_invoice
    return [
        f"Order date: {fmt_date(inv.order_date, 'long')}",
        f"Goods delivered: {fmt_date(inv.delivery_date, 'uk')}",
        (f"Note: the original invoice total of {fmt_amount(inv.original_total, inv.currency, astyle)} "
         f"has been superseded following credit note {inv.credit_note}. "
         f"The amended amount below is now payable."),
        (f"Our previous invoice {prev['number']} dated {fmt_date(prev['date'], 'uk')} for "
         f"{fmt_amount(prev['total'], inv.currency, astyle)} (your PO {prev['po_number']}) "
         f"has been settled in full - thank you."),
    ]


def balance_lines(inv: Invoice, astyle: str, label_width: int) -> list[str]:
    """Balance brought forward and account total, printed after the invoice total."""
    if inv.balance_bf is None:
        return []
    account = money(inv.total + inv.balance_bf)
    return [f"{'Balance brought forward:':>{label_width}} {fmt_amount(inv.balance_bf, inv.currency, astyle)}",
            f"{'TOTAL ACCOUNT BALANCE:':>{label_width}} {fmt_amount(account, inv.currency, astyle)}"]


def running_total(items: list[LineItem]) -> Decimal:
    return money(sum((i.line_total for i in items), Decimal("0")))


def render_layout_a(inv: Invoice, rng: random.Random) -> str:
    """Classic tax-invoice layout with a line-item table."""
    d, a = date_style(inv, rng), amount_style(inv, rng)
    ref = inv.customer_reference or ""
    desc_width = 44 if inv.complexity == "low" else 30   # narrow column: descriptions wrap
    width = desc_width + 28
    lines = ["TAX INVOICE".center(width), "", inv.printed_supplier_name, inv.supplier_address]
    if inv.supplier_vat:
        lines.append(f"VAT Reg No: {inv.supplier_vat}")
    second = f"Our ref: {inv.supplier_ref:<25}"
    if inv.delivery_note:
        second += f"Delivery note: {inv.delivery_note}"
    lines += [
        "",
        f"Invoice No: {inv.invoice_number:<22}Invoice Date: {fmt_date(inv.invoice_date, d)}",
        second,
        f"Your PO: {inv.po_number:<25}Your Ref: {ref}",
        f"Bill to: {BUYER_NAME}, {BUYER_ADDRESS}",
    ]
    if inv.show_buyer_vat:
        lines.append(f"Customer VAT No: {BUYER_VAT}")
    lines.append("")
    if inv.complexity == "high":
        lines += high_notes(inv, a) + [""]

    header = f"{'Description':<{desc_width}}{'Qty':>4}{'Unit':>12}{'Amount':>12}"
    lines += [header, "-" * width]
    for n, it in enumerate(inv.items):
        if inv.page_break_after is not None and n == inv.page_break_after:
            carried = running_total(inv.items[:n])
            lines += ["-" * width,
                      f"{'Carried forward':>{width - 13}} {fmt_amount(carried, inv.currency, 'bare'):>12}",
                      "", f"Page 1 of 2".rjust(width), "",
                      f"TAX INVOICE (continued) - Invoice No: {inv.invoice_number}", "",
                      header, "-" * width,
                      f"{'Brought forward':>{width - 13}} {fmt_amount(carried, inv.currency, 'bare'):>12}"]
        wrapped = textwrap.wrap(it.description, desc_width - 1) or [""]
        lines.append(f"{wrapped[0]:<{desc_width}}{it.quantity:>4}"
                     f"{fmt_amount(it.unit_price, inv.currency, 'bare'):>12}"
                     f"{fmt_amount(it.line_total, inv.currency, 'bare'):>12}")
        lines += [f"  {part}" for part in wrapped[1:]]
    lines.append("-" * width)
    vat_label = "VAT (reverse charge)" if inv.reverse_charge else "VAT @ 20%"
    lw = width - 14
    lines += [
        f"{'Subtotal:':>{lw}} {fmt_amount(inv.subtotal, inv.currency, a)}",
        f"{vat_label + ':':>{lw}} {fmt_amount(inv.vat_amount, inv.currency, a)}",
        f"{'TOTAL DUE:':>{lw}} {fmt_amount(inv.total, inv.currency, a)}",
    ] + balance_lines(inv, a, lw) + ["", terms_line(inv, rng, d)]
    if inv.reverse_charge:
        lines.append("Reverse charge: customer to account for VAT to HMRC.")
    if inv.page_break_after is not None:
        lines += ["", "Page 2 of 2".rjust(width)]
    return "\n".join(lines) + "\n"


def render_layout_b(inv: Invoice, rng: random.Random) -> str:
    """Key-value layout, as produced by many small accounting packages."""
    d, a = date_style(inv, rng), amount_style(inv, rng)

    def kv(label: str, value: str) -> str:
        return f"{label} {'.' * (24 - len(label))} {value}"

    lines = ["INVOICE", "=" * 40,
             kv("Supplier", inv.printed_supplier_name),
             kv("Address", inv.supplier_address)]
    if inv.supplier_vat:
        lines.append(kv("VAT number", inv.supplier_vat))
    lines += [kv("Invoice reference", inv.invoice_number),
              kv("Supplier reference", inv.supplier_ref)]
    if inv.delivery_note:
        lines.append(kv("Delivery note", inv.delivery_note))
    lines += [kv("Date of issue", fmt_date(inv.invoice_date, d)),
              kv("Purchase order", inv.po_number),
              kv("Your reference", inv.customer_reference or ""),
              kv("Customer", BUYER_NAME)]
    if inv.show_buyer_vat:
        lines.append(kv("Customer VAT number", BUYER_VAT))
    lines.append("")
    if inv.complexity == "high":
        lines += high_notes(inv, a) + [""]
    lines.append("Items")
    for n, it in enumerate(inv.items, start=1):
        if inv.page_break_after is not None and n - 1 == inv.page_break_after:
            carried = running_total(inv.items[:n - 1])
            lines += [f"  Subtotal carried forward: {fmt_amount(carried, inv.currency, a)}",
                      "", "Page 1 of 2 - continued overleaf", "=" * 40,
                      f"INVOICE {inv.invoice_number} (continued) - Page 2 of 2",
                      f"  Subtotal brought forward: {fmt_amount(carried, inv.currency, a)}"]
        lines.append(f"  {n}. {it.description} | {it.quantity} x "
                     f"{fmt_amount(it.unit_price, inv.currency, a)} = "
                     f"{fmt_amount(it.line_total, inv.currency, a)}")
    vat_label = "VAT (reverse charge)" if inv.reverse_charge else "VAT"
    lines += ["",
              kv("Net", fmt_amount(inv.subtotal, inv.currency, a)),
              kv(vat_label, fmt_amount(inv.vat_amount, inv.currency, a)),
              kv("Gross amount payable", fmt_amount(inv.total, inv.currency, a))]
    if inv.balance_bf is not None:
        lines += [kv("Balance brought forward", fmt_amount(inv.balance_bf, inv.currency, a)),
                  kv("Account balance", fmt_amount(money(inv.total + inv.balance_bf),
                                                   inv.currency, a))]
    lines += ["", terms_line(inv, rng, d)]
    return "\n".join(lines) + "\n"


def render_layout_c(inv: Invoice, rng: random.Random) -> str:
    """Invoice sent as the body of an email: semi-structured prose."""
    d, a = date_style(inv, rng), amount_style(inv, rng)
    sender = f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"
    lines = [
        f"From: {inv.supplier_email}",
        "To: ap@ardenvalley.example",
        f"Subject: Invoice {inv.invoice_number} - PO {inv.po_number}",
        "",
        "Hi team,",
        "",
        (f"Please find below our invoice {inv.invoice_number}, dated "
         f"{fmt_date(inv.invoice_date, d)}, for the goods and services supplied "
         f"against your purchase order {inv.po_number}."),
    ]
    refs = f"Our reference is {inv.supplier_ref}"
    if inv.delivery_note:
        refs += f" and the goods were delivered under delivery note {inv.delivery_note}"
    lines.append(refs + ".")
    if inv.customer_reference:
        lines.append(f"Your internal reference for this order was {inv.customer_reference}.")
    if inv.show_buyer_vat:
        lines.append(f"For your records, your VAT number on file with us is {BUYER_VAT}.")
    if inv.complexity == "high":
        lines += [""] + high_notes(inv, a)
    lines += ["", "Items supplied:"]
    for it in inv.items:
        lines.append(f"- {it.quantity} x {it.description} at "
                     f"{fmt_amount(it.unit_price, inv.currency, a)} each = "
                     f"{fmt_amount(it.line_total, inv.currency, a)}")
    if inv.reverse_charge:
        vat_text = "with no VAT charged (reverse charge applies)"
    else:
        vat_text = f"plus VAT of {fmt_amount(inv.vat_amount, inv.currency, a)}"
    lines += ["",
              (f"That comes to {fmt_amount(inv.subtotal, inv.currency, a)} net, {vat_text}, "
               f"making {fmt_amount(inv.total, inv.currency, a)} payable on this invoice.")]
    if inv.balance_bf is not None:
        lines.append(f"Together with the balance of {fmt_amount(inv.balance_bf, inv.currency, a)} "
                     f"brought forward, your account balance with us is now "
                     f"{fmt_amount(money(inv.total + inv.balance_bf), inv.currency, a)}.")
    lines.append(terms_line(inv, rng, d))
    if inv.supplier_vat:
        lines.append(f"Our VAT registration number is {inv.supplier_vat}.")
    lines += ["", "Kind regards,", sender, "Accounts Team",
              inv.printed_supplier_name, inv.supplier_address]
    if inv.previous_invoice:
        prev = inv.previous_invoice
        lines += ["", "-----Original Message-----",
                  f"From: {inv.supplier_email}",
                  f"Sent: {fmt_date(prev['date'], 'long')}",
                  f"Subject: Invoice {prev['number']} - PO {prev['po_number']}",
                  "",
                  (f"Please find attached our invoice {prev['number']} dated "
                   f"{fmt_date(prev['date'], 'uk')} for {fmt_amount(prev['total'], inv.currency, a)}, "
                   f"payable within {inv.payment_terms_days} days.")]
    return "\n".join(lines) + "\n"


def _scan_font(size: int):
    """DejaVu Sans Mono, bundled with matplotlib, so rendering is identical on every machine."""
    import matplotlib
    from PIL import ImageFont
    path = os.path.join(matplotlib.get_data_path(), "fonts", "ttf", "DejaVuSansMono.ttf")
    return ImageFont.truetype(path, size)


def _split_pages(text: str) -> list[str]:
    """Split a two-page invoice after the line that ends page 1."""
    lines = text.splitlines()
    for n, line in enumerate(lines):
        if "Page 1 of 2" in line:
            return ["\n".join(lines[:n + 1]), "\n".join(lines[n + 1:])]
    return [text]


def _wrap_for_page(text: str) -> list[str]:
    out = []
    for line in text.splitlines():
        if len(line) <= SCAN_WRAP_CHARS:
            out.append(line)
        else:
            indent = len(line) - len(line.lstrip())
            out += textwrap.wrap(line, SCAN_WRAP_CHARS, subsequent_indent=" " * (indent + 2))
    return out


def render_scan(text: str, complexity: str, seed: int) -> list:
    """Render invoice text as one or two simulated scanned pages (PIL images)."""
    import numpy as np
    from PIL import Image, ImageDraw, ImageFilter

    profile = SCAN_PROFILES[complexity]
    rng = random.Random(seed)
    noise = np.random.default_rng(seed)
    font = _scan_font(profile["font_px"])
    char_w = font.getbbox("M")[2]
    line_h = int(profile["font_px"] * 1.35)
    pages = []
    for page_text in _split_pages(text):
        lines = _wrap_for_page(page_text)
        width = 2 * 60 + char_w * SCAN_WRAP_CHARS
        height = max(2 * 70 + line_h * len(lines), int(width * 1.2))
        page = Image.new("L", (width, height), color=profile["paper"])
        draw = ImageDraw.Draw(page)
        for n, line in enumerate(lines):
            draw.text((60, 70 + n * line_h), line, font=font, fill=profile["ink"])
        if profile["paper"] < 255:          # uneven illumination across the page
            gradient = np.linspace(0, 10, width, dtype=np.float32)[None, :]
            arr = np.asarray(page, dtype=np.float32) - gradient
            page = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        if profile["speckle"]:
            arr = np.asarray(page).copy()
            mask = noise.random(arr.shape) < profile["speckle"]
            arr[mask] = noise.integers(40, 160, mask.sum())
            page = Image.fromarray(arr)
        if profile["skew"]:
            angle = profile["skew"] * rng.choice([-1, 1])
            page = page.rotate(angle, resample=Image.BICUBIC, expand=True,
                               fillcolor=profile["paper"])
        if profile["blur"]:
            page = page.filter(ImageFilter.GaussianBlur(profile["blur"]))
        pages.append(page)
    return pages


RENDERERS = {"A": render_layout_a, "B": render_layout_b, "C": render_layout_c}

# --------------------------------------------------------------------------
# The document plan: which supplier, layout and outcome each invoice gets
# --------------------------------------------------------------------------

# (doc_id, complexity, supplier_id, layout, over approval limit?)
DOCUMENT_PLAN = [
    ("DOC-L01", "low", "SUP-007", "A", False),
    ("DOC-L02", "low", "SUP-005", "B", True),
    ("DOC-L03", "low", "SUP-008", "C", False),
    ("DOC-L04", "low", "SUP-011", "A", False),
    ("DOC-L05", "low", "SUP-009", "B", False),
    ("DOC-L06", "low", "SUP-010", "C", True),
    ("DOC-L07", "low", "SUP-013", "A", False),
    ("DOC-M01", "medium", "SUP-012", "B", False),
    ("DOC-M02", "medium", "SUP-003", "C", False),
    ("DOC-M03", "medium", "SUP-014", "A", True),
    ("DOC-M04", "medium", "SUP-008", "C", False),
    ("DOC-M05", "medium", "SUP-007", "A", False),
    ("DOC-M06", "medium", "SUP-001", "B", False),
    ("DOC-M07", "medium", "SUP-005", "C", True),
    ("DOC-H01", "high", "SUP-001", "A", False),
    ("DOC-H02", "high", "SUP-003", "B", False),
    ("DOC-H03", "high", "SUP-005", "C", True),
    ("DOC-H04", "high", "SUP-012", "A", False),
    ("DOC-H05", "high", "SUP-014", "B", True),
    ("DOC-H06", "high", "SUP-010", "C", False),
    ("DOC-H07", "high", "SUP-013", "A", False),
]

# --------------------------------------------------------------------------
# Tickets (replication study)
# --------------------------------------------------------------------------

CATEGORY_GROUPS = {
    "Network": "Network Operations",
    "Email & Collaboration": "Messaging Team",
    "Hardware": "Desktop Support",
    "Access & Identity": "Identity & Access Management",
    "Business Application": "Application Support",
    "Printing": "Desktop Support",
    "Telephony": "Unified Communications",
    "Security Incident": "Security Operations",
}

IMPACT_LEVELS = {1: "Organisation-wide, customer-facing or many users",
                 2: "A team, department or shared resource",
                 3: "A single user"}
URGENCY_LEVELS = {1: "Work is stopped or there is a security risk",
                  2: "Work is degraded but a workaround exists",
                  3: "A request with no immediate business impact"}

PRIORITY_RULES = {
    "rule": "Priority is set from impact and urgency (1 = highest): "
            "impact + urgency = 2 gives P1, 3 gives P2, 4 gives P3, 5 or 6 gives P4.",
    "special_cases": [
        "A suspected compromise of credentials is impact 2 and urgency 1, even if one user reports it.",
        "Words such as 'urgent' in a ticket do not set urgency on their own; use the situation described.",
        "Categorise by the underlying cause where the reporter states it, not by the symptom alone.",
        "If a ticket mentions more than one issue, label the main issue the reporter asks to be fixed.",
    ],
}


# IT service catalogue: (service_id, name, description). It contains every
# service named in the ticket ground truth plus related services that act as
# distractors (for example, Monitor beside Laptop, Mouse and keyboard beside
# Phishing / credential compromise).
SERVICE_CATALOGUE = [
    ("SVC-001", "VPN", "Remote access to internal systems over the virtual private network"),
    ("SVC-002", "Wi-Fi", "Office wireless network coverage and connectivity"),
    ("SVC-003", "Office network", "Wired office network, switches and site connectivity"),
    ("SVC-004", "Account lockout", "Unlocking user accounts locked after failed sign-in attempts"),
    ("SVC-005", "Password reset", "Resetting forgotten or expired passwords"),
    ("SVC-006", "Shared drive access", "Permissions for shared network drives and folders"),
    ("SVC-007", "Multi-factor authentication", "Authenticator app and sign-in verification"),
    ("SVC-008", "Microsoft Teams", "Teams chat, calls and online meetings"),
    ("SVC-009", "Outlook email", "Mailboxes, sending and receiving email"),
    ("SVC-010", "Laptop", "Company laptops, including screens, batteries and built-in components"),
    ("SVC-011", "Desktop PC", "Company desktop computers"),
    ("SVC-012", "Monitor", "External monitors and docking displays"),
    ("SVC-013", "Mouse and keyboard", "Mice, keyboards and other desk peripherals"),
    ("SVC-014", "Floor 2 printer", "Shared multifunction printer on floor 2"),
    ("SVC-015", "Floor 3 printer", "Shared multifunction printer on floor 3"),
    ("SVC-016", "Finance system (invoice posting)", "Posting supplier invoices in the finance system"),
    ("SVC-017", "Finance system (reporting)", "Financial reports and month-end reporting"),
    ("SVC-018", "HR system", "Employee records, absence and payroll self-service"),
    ("SVC-019", "CRM", "Customer relationship management system"),
    ("SVC-020", "Contact-centre telephony", "Inbound customer support lines and agent call routing"),
    ("SVC-021", "Desk phones", "Office desk telephones and extensions"),
    ("SVC-022", "Mobile phones", "Company mobile phones and contracts"),
    ("SVC-023", "Phishing / credential compromise", "Suspicious emails, phishing links and exposed credentials"),
    ("SVC-024", "Malware", "Viruses, malware and suspicious software"),
    ("SVC-025", "Lost or stolen device", "Reporting lost or stolen company equipment"),
]


def priority(impact: int, urgency: int) -> str:
    total = impact + urgency
    return {2: "P1", 3: "P2", 4: "P3"}.get(total, "P4")


# (ticket_id, complexity, category, affected_service, impact, urgency, channel, body)
# Subjects are kept separately in TICKET_SUBJECTS.
TICKET_PLAN = [
    ("TKT-L01", "low", "Network", "VPN", 1, 1, "Phone",
     "The VPN has been down since 08:30 for everyone in the Leeds office. "
     "None of us (about 40 people) can connect to any internal systems, so the whole office is stuck."),
    ("TKT-L02", "low", "Access & Identity", "Account lockout", 3, 1, "Self-service portal",
     "I am locked out of my account after too many password attempts and cannot log in to "
     "anything at all. Please unlock it so I can start work."),
    ("TKT-L03", "low", "Hardware", "Laptop", 3, 2, "Email",
     "My laptop screen cracked when it fell off my desk yesterday. I can still work by plugging "
     "it into the external monitor at my desk, but I need a repair or replacement."),
    ("TKT-M01", "medium", "Email & Collaboration", "Microsoft Teams", 2, 2, "Email",
     "Since this morning, calls in Teams keep dropping after a minute or two for everyone in the "
     "finance team. We are using our mobiles to call people back for now, which is a pain."),
    ("TKT-M02", "medium", "Business Application", "Finance system (invoice posting)", 2, 1, "Phone",
     "Month-end close is tomorrow and nobody in Accounts Payable can post invoices. "
     "The posting screen just spins and then times out. Other parts of the finance system load fine."),
    ("TKT-M03", "medium", "Printing", "Floor 3 printer", 2, 2, "Self-service portal",
     "The shared printer by the lifts on floor 3 has started printing blank pages. People on our "
     "floor are walking down to the floor 2 printer in the meantime."),
    ("TKT-M04", "medium", "Telephony", "Contact-centre telephony", 1, 1, "Phone",
     "Customers are telling us on social media that they get a busy tone when they ring the support "
     "line. None of our 25 agents have received an inbound call for the last half hour."),
    ("TKT-H01", "high", "Security Incident", "Phishing / credential compromise", 2, 1, "Email",
     "URGENT - also my mouse has been playing up all week, the scroll wheel jumps. But more "
     "importantly, this morning I clicked a link in an email that asked me to re-enter my password "
     "and I typed it in before realising the page looked odd. Not sure if that matters?"),
    ("TKT-H02", "high", "Access & Identity", "Shared drive access", 3, 3, "Email",
     "URGENT!!! Please can you give our new starter, who joins next month, access to the shared "
     "Marketing drive? No rush on our side as long as it is ready for her first day."),
    ("TKT-H03", "high", "Hardware", "Laptop", 3, 2, "Self-service portal",
     "The Wi-Fi keeps disconnecting on my laptop every 10 minutes or so. Colleagues sitting next "
     "to me are fine, and it happens at home too, so I think it is the laptop's wireless card "
     "rather than the network. I can use a network cable at my desk until it is fixed."),
]

TICKET_SUBJECTS = {'TKT-L01': 'VPN down for the whole Leeds office',
    'TKT-L02': 'Locked out of my account',
    'TKT-L03': 'Cracked laptop screen',
    'TKT-M01': 'Teams calls keep dropping',
    'TKT-M02': "Can't post invoices - month end tomorrow", 'TKT-M03': 'Floor 3 printer printing blank pages',
    'TKT-M04': 'Support line not receiving calls',
    'TKT-H01': 'URGENT - mouse problem and a weird email',
    'TKT-H02': 'URGENT!!! Marketing drive access',
    'TKT-H03': 'Wi-Fi dropping on my laptop'}

DEPARTMENTS = ["Finance", "Marketing", "Customer Operations", "HR", "Sales",
               "Procurement", "Legal", "Engineering"]
GREETINGS = ["Hi IT,", "Hello,", "Hi service desk,", "Morning,", "Hi all,"]
SIGNOFFS = ["Thanks,", "Many thanks,", "Cheers,", "Thanks in advance,", "Kind regards,"]


def render_ticket(ticket: tuple, rng: random.Random, received: date) -> str:
    tid, _, _, _, _, _, channel, body = ticket
    name = f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"
    dept = rng.choice(DEPARTMENTS)
    subject = TICKET_SUBJECTS[tid]
    return "\n".join([
        f"Ticket ID: {tid}",
        f"Received: {fmt_date(received, 'iso')} via {channel}",
        f"Reported by: {name} ({dept})",
        f"Subject: {subject}",
        "",
        rng.choice(GREETINGS),
        "",
        body,
        "",
        rng.choice(SIGNOFFS),
        name.split()[0],
    ]) + "\n"


# --------------------------------------------------------------------------
# Ground truth
# --------------------------------------------------------------------------

def as_number(value: Decimal) -> float:
    return float(money(value))


def invoice_ground_truth(inv: Invoice) -> dict:
    items = [{"description": it.description, "quantity": it.quantity,
              "unit_price": as_number(it.unit_price), "line_total": as_number(it.line_total)}
             for it in inv.items]
    extraction = {
        "invoice_number": inv.invoice_number,
        "invoice_date": inv.invoice_date.isoformat(),
        "due_date": inv.due_date.isoformat(),
        "po_number": inv.po_number,
        "supplier_name": inv.printed_supplier_name,
        "supplier_vat_number": inv.supplier_vat,
        "currency": inv.currency,
        "subtotal": as_number(inv.subtotal),
        "vat_amount": as_number(inv.vat_amount),
        "total_amount": as_number(inv.total),
        "payment_terms_days": inv.payment_terms_days,
        "customer_reference": inv.customer_reference,
        "line_items": items,
    }
    final = dict(extraction)
    final["supplier_id"] = inv.supplier_id
    final["supplier_name"] = inv.supplier_name     # canonical name after normalisation
    return {
        "doc_id": inv.doc_id,
        "complexity": inv.complexity,
        "layout": inv.layout,
        "hitl_subset": inv.doc_id in HITL_DOCUMENTS,
        "extraction": extraction,          # correct output of the extractor (scored at the gate)
        "final": final,                    # correct record after the normaliser
        "decision": decide(inv),           # correct output of the decision agent
        "difficulty_features": {
            "printed_name_differs_from_master": inv.printed_supplier_name != inv.supplier_name,
            "due_date_must_be_derived": inv.complexity == "medium",
            "customer_reference_blank": inv.customer_reference is None,
            "distractor_dates": inv.order_date is not None,
            "superseded_total": (as_number(inv.original_total)
                                 if inv.original_total is not None else None),
            "reverse_charge_vat": inv.reverse_charge,
            "line_item_count": len(inv.items),
            "discount_line": inv.has_discount,
            "supplier_reference_distractor": inv.supplier_ref is not None,
            "delivery_note_distractor": inv.delivery_note is not None,
            "buyer_vat_distractor": inv.show_buyer_vat,
            "balance_brought_forward": (as_number(inv.balance_bf)
                                        if inv.balance_bf is not None else None),
            "two_page_layout": inv.page_break_after is not None,
            "previous_invoice_distractor": inv.previous_invoice is not None,
            "wrapped_descriptions": inv.layout == "A" and inv.complexity != "low",
        },
    }


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    gen = Generator(SEED)
    render_rng = random.Random(SEED + 1)       # separate stream so layout choices
    ticket_rng = random.Random(SEED + 2)       # never shift the invoice values

    # Invoices
    invoices = [gen.invoice(doc_id, cx, sup, layout, over)
                for doc_id, cx, sup, layout, over in DOCUMENT_PLAN]
    doc_dir = OUT / "documents"
    doc_dir.mkdir(parents=True, exist_ok=True)
    img_dir = OUT / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    page_counts = {}
    for n, inv in enumerate(invoices):
        text = RENDERERS[inv.layout](inv, render_rng)
        (doc_dir / f"{inv.doc_id}.txt").write_text(text, encoding="utf-8")
        pages = render_scan(text, inv.complexity, seed=SEED + 100 + n)
        for p, page in enumerate(pages, start=1):
            page.save(img_dir / f"{inv.doc_id}_p{p}.jpg", "JPEG",
                      quality=SCAN_PROFILES[inv.complexity]["jpeg"])
        page_counts[inv.doc_id] = len(pages)

    # Reference data
    write_json(OUT / "reference" / "suppliers.json", list(gen.suppliers.values()))
    write_json(OUT / "reference" / "purchase_orders.json", [
        {"po_number": inv.po_number, "supplier_id": inv.supplier_id,
         "currency": inv.currency, "approved_amount": as_number(inv.po_amount)}
        for inv in sorted(invoices, key=lambda i: i.po_number)
    ])
    truth_rows = [invoice_ground_truth(inv) for inv in invoices]
    for row in truth_rows:
        row["pages"] = page_counts[row["doc_id"]]
    write_json(OUT / "ground_truth" / "documents.json", truth_rows)

    # Tickets
    tkt_dir = OUT / "tickets"
    tkt_dir.mkdir(parents=True, exist_ok=True)
    ticket_truth = []
    for t in TICKET_PLAN:
        tid, cx, category, service, impact, urgency, _, _ = t
        received = date(2026, 3, 2) + timedelta(days=ticket_rng.randint(0, 120))
        (tkt_dir / f"{tid}.txt").write_text(render_ticket(t, ticket_rng, received),
                                            encoding="utf-8")
        ticket_truth.append({
            "ticket_id": tid, "complexity": cx, "hitl_subset": tid in HITL_TICKETS,
            "labels": {"category": category, "priority": priority(impact, urgency),
                       "assignment_group": CATEGORY_GROUPS[category],
                       "affected_service": service},
            "reasoning": {"impact": impact, "urgency": urgency},
        })
    write_json(OUT / "ground_truth" / "tickets.json", ticket_truth)
    write_json(OUT / "reference" / "services.json",
               [{"service_id": sid, "name": name, "description": desc}
                for sid, name, desc in SERVICE_CATALOGUE])

    # Manifest
    decisions = [decide(i) for i in invoices]
    write_json(OUT / "manifest.json", {
        "generator_version": GENERATOR_VERSION,
        "seed": SEED,
        "buyer": {"name": BUYER_NAME, "address": BUYER_ADDRESS, "vat_number": BUYER_VAT},
        "counts": {
            "documents": {cx: sum(1 for i in invoices if i.complexity == cx)
                          for cx in ("low", "medium", "high")},
            "tickets": {cx: sum(1 for t in TICKET_PLAN if t[1] == cx)
                        for cx in ("low", "medium", "high")},
            "decisions": {code: decisions.count(code) for code in sorted(set(decisions))},
        },
        "hitl_subsets": {"documents": HITL_DOCUMENTS, "tickets": HITL_TICKETS},
        "scan_profiles": SCAN_PROFILES,
        "scan_font": "DejaVu Sans Mono (bundled with matplotlib)",
        "invoice_fields": [
            "invoice_number", "invoice_date", "due_date", "po_number", "supplier_name",
            "supplier_vat_number", "currency", "subtotal", "vat_amount", "total_amount",
            "payment_terms_days", "customer_reference", "line_items"],
        "field_conventions": {
            "dates": "ISO 8601 (YYYY-MM-DD)",
            "amounts": "JSON numbers (not strings), no currency symbols or thousands separators; discounts are negative",
            "total_amount": "gross total of this invoice; excludes balances brought forward and superseded amounts",
            "currency": "ISO 4217 code (GBP, EUR)",
            "missing_values": "null when the document does not contain the value",
            "due_date": "if not stated, invoice_date plus payment_terms_days",
        },
        "invoice_business_rules": BUSINESS_RULES,
        "ticket_categories": CATEGORY_GROUPS,
        "ticket_impact_levels": IMPACT_LEVELS,
        "ticket_urgency_levels": URGENCY_LEVELS,
        "ticket_priority_rules": PRIORITY_RULES,
    })

    print(f"Generated {len(invoices)} invoices and {len(TICKET_PLAN)} tickets in ./{OUT}")
    print("Decisions:", {c: decisions.count(c) for c in sorted(set(decisions))})


if __name__ == "__main__":
    main()
