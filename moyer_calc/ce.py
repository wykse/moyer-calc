from pathlib import Path
import pandas as pd
import numpy as np
from attrs import define, field


EMISSIONS_TABLE = pd.read_csv(Path(__file__).parents[1] / "output/emission_factors.csv")


@define
class Engine:
    id: str
    engine_type: str
    year: int
    hp: int
    load_factor: float
    emission_standard: str
    emission_factors: pd.DataFrame = field(init=False)

    def __attrs_post_init__(self):
        self.emission_factors = EMISSIONS_TABLE.loc[
            (EMISSIONS_TABLE["engine_type"] == self.engine_type)
            & (EMISSIONS_TABLE["hp_min"] <= self.hp)
            & (EMISSIONS_TABLE["hp_max"] >= self.hp)
            & (EMISSIONS_TABLE["standard"] == self.emission_standard)
            & (EMISSIONS_TABLE["model_year_min"] <= self.year)
            & (EMISSIONS_TABLE["model_year_max"] >= self.year)
        ]

        # TODO: add check for emission factor return

    def calc_annual_emissions(
        self,
        project_life: int,
        start_year: int,
        annual_activity: int,
        percent_operation: float,
        baseline: bool,
    ):
        # Determine deterioration life
        if baseline is True:
            det_life = start_year - self.year + (project_life / 2)
        else:
            det_life = project_life / 2

        total_equip_act = annual_activity * det_life

        # Get engine type, ci or lsi
        engine_type = self.engine_type.split("-")[0]

        # Determine total equipment activity
        match engine_type:
            case "ci":
                if total_equip_act > 12000:
                    total_equip_act = 12000
            case "lsi":
                if self.year <= 2006 and total_equip_act > 3500:
                    total_equip_act = 3500
                if self.year >= 2007 and total_equip_act > 5000:
                    total_equip_act = 5000
            case _:
                pass

        df = self.emission_factors.copy()
        df["total_equip_act"] = total_equip_act
        df["det_prod"] = df["dr"] * df["total_equip_act"]
        df["annual_emissions"] = (
            (df["ef"] + df["det_prod"])
            * self.hp
            * self.load_factor
            * annual_activity
            * percent_operation
            / 907200
        )

        df = df.set_index("pollutant")

        return df["annual_emissions"]


@define
class Equipment:
    id: str
    engine: Engine


@define
class SurplusEmissions:
    baseline_engine: list[Engine]
    reduced_engine: Engine
    baseline_count: int
    annual_activity: int
    percent_operation: float
    project_life: int
    baseline_emissions_nox: list[float]
    baseline_emissions_rog: list[float]
    baseline_emissions_pm: list[float]
    reduced_emissions_nox: float
    reduced_emissions_rog: float
    reduced_emissions_pm: float
    surplus_nox: float
    surplus_rog: float
    surplus_pm: float
    weighted: float
    unweighted: float
    percent_nox_reduced: float


@define
class SurplusEmissionsTwoStep:
    surplus_emissions_step_1: SurplusEmissions
    surplus_emissions_step_2: SurplusEmissions
    surplus_nox: float
    surplus_rog: float
    surplus_pm: float
    weighted: float
    unweighted: float


@define
class ProjectAlt:
    baseline_engine: list[Engine]
    reduced_engine: Engine
    percent_nox_red: float
    nox: float
    rog: float
    pm: float
    weighted: float
    rate: float
    annual_activity: list[float]
    project_life: int
    pot_grant_cel: int
    pot_grant_inc: int
    grant: int
    ce_per_ton: float
    grant_cel_minus_inc: float
    grant_dist: float
    total_activity: float
    percent_cost: float
    cel: float


def calc_crf(i: float, n: int) -> float:
    """Calculate capital recovery factor given interest rate and years.

    Args:
        i (float): Interest rate.
        n (int): Number of years.

    Returns:
        float: Capital recovery factor.
    """
    return (i * (1 + i) ** n) / ((1 + i) ** n - 1)


def round_down_hundred(number: int | float):
    """Round number down to the nearest hundred.

    Args:
        number (int | float): Number to round.

    Returns:
        int: Rounded number down to the nearest hundred.
    """
    return number - (number % 100)


def calc_pot_grant_cel_2s(
    cel_base,
    cel_at,
    surplus_base,
    surplus_at,
    project_life_step_1,
    project_life_step_2,
    i=0.01,
):
    s1 = cel_base * surplus_base / calc_crf(i, project_life_step_1)
    s2 = cel_at * surplus_at / calc_crf(i, project_life_step_2)
    return s1 + s2


def calc_surplus_emissions(
    baseline_engine: list[Engine],
    reduced_engine: Engine,
    start_year: int,
    annual_activity: list[int],
    percent_operation: list[int],
    project_life: int,
) -> SurplusEmissions:
    sum_activity = sum(annual_activity)
    wgt_percent_op = np.matmul(annual_activity, percent_operation) / sum_activity

    # Initialize an empty dataframe to hold baseline engine emissions
    base_emi = reduced_engine.calc_annual_emissions(
        project_life=0,
        start_year=0,
        annual_activity=0,
        percent_operation=0,
        baseline=True,
    )

    temp_base_nox = []
    temp_base_rog = []
    temp_base_pm = []

    # Loop through baseline engine and sum emissions
    for i, engine in enumerate(baseline_engine):
        temp_emi = engine.calc_annual_emissions(
            project_life=project_life,
            start_year=start_year,
            annual_activity=annual_activity[i],
            percent_operation=percent_operation[i],
            baseline=True,
        )
        temp_base_nox.append(temp_emi["nox"])
        temp_base_rog.append(temp_emi["rog"])
        temp_base_pm.append(temp_emi["pm"])

        base_emi += temp_emi

    red_emi = reduced_engine.calc_annual_emissions(
        project_life=project_life,
        start_year=start_year,
        annual_activity=sum_activity,
        percent_operation=wgt_percent_op,
        baseline=False,
    )

    surplus_emi = base_emi - red_emi

    emission_reductions = SurplusEmissions(
        baseline_engine=baseline_engine,
        reduced_engine=reduced_engine,
        baseline_count=len(baseline_engine),
        annual_activity=sum_activity,
        percent_operation=wgt_percent_op,
        project_life=project_life,
        baseline_emissions_nox=temp_base_nox,
        baseline_emissions_rog=temp_base_rog,
        baseline_emissions_pm=temp_base_pm,
        reduced_emissions_nox=red_emi["nox"],
        reduced_emissions_rog=red_emi["rog"],
        reduced_emissions_pm=red_emi["pm"],
        surplus_nox=surplus_emi["nox"],
        surplus_rog=surplus_emi["rog"],
        surplus_pm=surplus_emi["pm"],
        weighted=surplus_emi["nox"] + surplus_emi["rog"] + (20 * surplus_emi["pm"]),
        unweighted=surplus_emi["nox"] + surplus_emi["rog"] + surplus_emi["pm"],
        percent_nox_reduced=(base_emi["nox"] - red_emi["nox"]) / base_emi["nox"],
    )

    return emission_reductions


def calc_surplus_emissions_2s(
    baseline_engine: list[Engine],
    reduced_engine: Engine,
    start_year: int,
    annual_activity: list[int],
    percent_operation: list[int],
    project_life_step_1: int,
    project_life_step_2: int,
):
    # Current emission standards for new engines
    CURRENT_STANDARD = {"ci": "t4f", "lsi-g": "c", "lsi-a": "c"}

    # Get lowest hp
    min_hp = min([engine.hp for engine in baseline_engine])

    # Get lowest load factor
    min_load_factor = min([engine.load_factor for engine in baseline_engine])

    # Get engine type for step 1; pick based on cleanest
    # LSI to CI is ineligible; assume lsi-a is cleaner than lsi-g
    if all(
        engine.engine_type == baseline_engine[0].engine_type
        for engine in baseline_engine
    ):
        engine_type_s1 = baseline_engine[0].engine_type
    elif "lsi-a" in [engine.engine_type for engine in baseline_engine]:
        engine_type_s1 = "lsi-a"
    else:
        engine_type_s1 = "lsi-g"

    # Step 1 reduced baseline information
    reduced_baseline_engine = Engine(
        id="Reduced Base",
        engine_type=engine_type_s1,
        year=start_year,
        hp=min_hp,
        load_factor=min_load_factor,
        emission_standard=CURRENT_STANDARD[engine_type_s1],
    )

    # Step 1 reduced baseline
    s1 = calc_surplus_emissions(
        baseline_engine=baseline_engine,
        reduced_engine=reduced_baseline_engine,
        start_year=start_year,
        annual_activity=annual_activity,
        percent_operation=percent_operation,
        project_life=project_life_step_1,
    )

    # Step 2 advanced technology
    s2 = calc_surplus_emissions(
        baseline_engine=[reduced_baseline_engine],
        reduced_engine=reduced_engine,
        start_year=start_year,
        annual_activity=[sum(annual_activity)],
        percent_operation=[
            np.matmul(annual_activity, percent_operation) / sum(annual_activity)
        ],
        project_life=project_life_step_2,
    )

    surplus = SurplusEmissionsTwoStep(
        surplus_emissions_step_1=s1,
        surplus_emissions_step_2=s2,
        surplus_nox=(s1.surplus_nox * s1.project_life / project_life_step_2)
        + (s2.surplus_nox * s2.project_life / project_life_step_2),
        surplus_rog=(s1.surplus_rog * s1.project_life / project_life_step_2)
        + (s2.surplus_rog * s2.project_life / project_life_step_2),
        surplus_pm=(s1.surplus_pm * s1.project_life / project_life_step_2)
        + (s2.surplus_pm * s2.project_life / project_life_step_2),
        weighted=(s1.weighted * s1.project_life / project_life_step_2)
        + (s2.weighted * s2.project_life / project_life_step_2),
        unweighted=(s1.unweighted * s1.project_life / project_life_step_2)
        + (s2.unweighted * s2.project_life / project_life_step_2),
    )

    return surplus


def min_annual_activity(
    baseline_engine: list[Engine],
    reduced_engine: Engine,
    start_year: int,
    annual_activity: list[int],
    percent_operation: list[int],
    project_life: int,
    ce_limit: int | float,
    cost_reduced_engine: int | float,
    max_percent: float,
    rate: float = 0.01,
    iterations: int = 20,
):
    surplus = calc_surplus_emissions(
        baseline_engine=baseline_engine,
        reduced_engine=reduced_engine,
        start_year=start_year,
        annual_activity=annual_activity,
        percent_operation=percent_operation,
        project_life=project_life,
    )

    crf = calc_crf(rate, project_life)
    pot_grant_cel = round_down_hundred(ce_limit * surplus.weighted / crf)
    pot_grant_inc = round_down_hundred(cost_reduced_engine * max_percent)
    grant = min(pot_grant_cel, pot_grant_inc)

    hold_alt = ProjectAlt(
        baseline_engine=baseline_engine,
        reduced_engine=reduced_engine,
        percent_nox_red=surplus.percent_nox_reduced,
        nox=surplus.surplus_nox,
        rog=surplus.surplus_rog,
        pm=surplus.surplus_pm,
        weighted=surplus.weighted,
        rate=rate,
        annual_activity=annual_activity,
        project_life=project_life,
        pot_grant_cel=pot_grant_cel,
        pot_grant_inc=pot_grant_inc,
        grant=min(pot_grant_cel, pot_grant_inc),
        ce_per_ton=min(pot_grant_cel, pot_grant_inc) * crf / surplus.weighted,
        grant_cel_minus_inc=pot_grant_cel - pot_grant_inc,
        grant_dist=abs(pot_grant_cel - pot_grant_inc),
        total_activity=np.array(annual_activity) * project_life,
        percent_cost=min(pot_grant_cel, pot_grant_inc) / cost_reduced_engine,
        cel=ce_limit,
    )

    # Limit iterations to prevent infinite loop
    n = 1

    # Check if potential grant at CE limit is greater than grant, else return
    if pot_grant_cel > grant:
        low_act = 0
        high_act = annual_activity
        bi_act = np.floor(np.add(low_act, high_act) / 2)
    else:
        return hold_alt

    # Hold bisection search annual activity and grant
    hold_bi_act = {}

    # Bisection search until maximize grant or exhaust iterations
    while n <= iterations:
        surplus_bi_act = calc_surplus_emissions(
            baseline_engine=baseline_engine,
            reduced_engine=reduced_engine,
            start_year=start_year,
            annual_activity=bi_act,
            percent_operation=percent_operation,
            project_life=project_life,
        )

        pot_grant_cel = round_down_hundred(ce_limit * surplus_bi_act.weighted / crf)

        hold_alt = ProjectAlt(
            baseline_engine=baseline_engine,
            reduced_engine=reduced_engine,
            percent_nox_red=surplus_bi_act.percent_nox_reduced,
            nox=surplus_bi_act.surplus_nox,
            rog=surplus_bi_act.surplus_rog,
            pm=surplus_bi_act.surplus_pm,
            weighted=surplus_bi_act.weighted,
            rate=rate,
            annual_activity=bi_act,
            project_life=project_life,
            pot_grant_cel=pot_grant_cel,
            pot_grant_inc=pot_grant_inc,
            grant=min(pot_grant_cel, pot_grant_inc),
            ce_per_ton=min(pot_grant_cel, pot_grant_inc)
            * crf
            / surplus_bi_act.weighted,
            grant_cel_minus_inc=pot_grant_cel - pot_grant_inc,
            grant_dist=abs(pot_grant_cel - pot_grant_inc),
            total_activity=np.array(annual_activity) * project_life,
            percent_cost=min(pot_grant_cel, pot_grant_inc) / cost_reduced_engine,
            cel=ce_limit,
        )

        print(n, low_act, high_act, bi_act, pot_grant_cel)
        hold_bi_act[pot_grant_cel] = bi_act

        if pot_grant_cel == pot_grant_inc:
            break

        n += 1

        if pot_grant_cel < grant:
            low_act = bi_act
        else:
            high_act = bi_act
        bi_act = np.floor(np.add(low_act, high_act) / 2)

    # If potential grant at CE limit is less than incremental grant, then
    # then bisection search went too far, go to activity that returns
    # potential grant at CE limit greater than incremental grant
    if pot_grant_cel < grant:
        pot_grant_cels = hold_bi_act.keys()
        pot_grant_cels = [g for g in pot_grant_cels if g >= grant]
        final_act = hold_bi_act[min(pot_grant_cels)]

        surplus_final_act = calc_surplus_emissions(
            baseline_engine=baseline_engine,
            reduced_engine=reduced_engine,
            start_year=start_year,
            annual_activity=final_act,
            percent_operation=percent_operation,
            project_life=project_life,
        )

        pot_grant_cel = round_down_hundred(ce_limit * surplus_final_act.weighted / crf)

        hold_alt = ProjectAlt(
            baseline_engine=baseline_engine,
            reduced_engine=reduced_engine,
            percent_nox_red=surplus_final_act.percent_nox_reduced,
            nox=surplus_final_act.surplus_nox,
            rog=surplus_final_act.surplus_rog,
            pm=surplus_final_act.surplus_pm,
            weighted=surplus_final_act.weighted,
            rate=rate,
            annual_activity=final_act,
            project_life=project_life,
            pot_grant_cel=pot_grant_cel,
            pot_grant_inc=pot_grant_inc,
            grant=min(pot_grant_cel, pot_grant_inc),
            ce_per_ton=min(pot_grant_cel, pot_grant_inc)
            * crf
            / surplus_final_act.weighted,
            grant_cel_minus_inc=pot_grant_cel - pot_grant_inc,
            grant_dist=abs(pot_grant_cel - pot_grant_inc),
            total_activity=np.array(annual_activity) * project_life,
            percent_cost=min(pot_grant_cel, pot_grant_inc) / cost_reduced_engine,
            cel=ce_limit,
        )

    return hold_alt


foo = Engine(
    id="1",
    engine_type="ci",
    year=2000,
    hp=120,
    load_factor=0.51,
    emission_standard="t0",
)
bar = Equipment(1, engine=foo)
# print(bar.engine.emission_factors)
# print(
#     bar.engine.calc_annual_emissions(
#         project_life=3,
#         start_year=2023,
#         annual_activity=200,
#         percent_operation=0.75,
#         baseline=True,
#     )
# )
# baz = bar.engine.calc_annual_emissions(
#     project_life=0,
#     start_year=0,
#     annual_activity=0,
#     percent_operation=0,
#     baseline=True,
# )

# print(baz)

red = Engine(
    id="1",
    engine_type="ci",
    year=2021,
    hp=120,
    load_factor=0.51,
    emission_standard="t4f",
)

surplus = calc_surplus_emissions(
    baseline_engine=[foo, foo],
    reduced_engine=red,
    start_year=2023,
    annual_activity=[500, 300],
    percent_operation=[1, 0.9],
    project_life=5,
)

# print(surplus)


engine_type = "ci"
emy = 2005
hp = 240
standard = "t2"
project_life = 3
year_1 = 2017
load_factor = 0.36
annual_activity = 750
percent_op = 1

red_engine_type = "ci"
red_hp = 210
red_standard = "t4f"
red_engine_my = 2017


base = Engine(
    id="1",
    engine_type=engine_type,
    year=emy,
    hp=hp,
    load_factor=load_factor,
    emission_standard=standard,
)


base2 = Engine(
    id="1",
    engine_type="ci",
    year=2004,
    hp=180,
    load_factor=load_factor,
    emission_standard="t2",
)


red = Engine(
    id="1",
    engine_type=red_engine_type,
    year=red_engine_my,
    hp=red_hp,
    load_factor=load_factor,
    emission_standard=red_standard,
)


surplus = calc_surplus_emissions(
    baseline_engine=[base, base2],
    reduced_engine=red,
    start_year=year_1,
    annual_activity=[533, 248],
    percent_operation=[percent_op, 1],
    project_life=project_life,
)


min_act = min_annual_activity(
    baseline_engine=[base, base2],
    reduced_engine=red,
    start_year=year_1,
    annual_activity=[annual_activity, 350],
    percent_operation=[percent_op, 1],
    project_life=3,
    ce_limit=30000,
    cost_reduced_engine=50000,
    max_percent=0.8,
    rate=0.01,
)


engine_type = "ci"
emy = 2006
hp = 503
standard = "t3"
project_life = 5
year_1 = 2017
load_factor = 0.73
annual_activity = 700
percent_op = 1

red_engine_type = "ci"
red_hp = 500
red_standard = "t4f"
red_engine_my = 2017

# base = OffRoadEquipment(
#     unit_id="Baseline",
#     engine_id="1",
#     engine_my=emy,
#     engine_type=engine_type,
#     hp=hp,
#     standard=standard,
#     emissions_table=EMISSIONS_TABLE,
# )

# red = OffRoadEquipment(
#     unit_id="Reduced",
#     engine_id="1",
#     engine_type=red_engine_type,
#     engine_my=red_engine_my,
#     hp=red_hp,
#     standard=red_standard,
#     emissions_table=EMISSIONS_TABLE,
# )

base = Engine(
    id="1",
    engine_type=engine_type,
    year=emy,
    hp=hp,
    load_factor=load_factor,
    emission_standard=standard,
)

red = Engine(
    id="1",
    engine_type=red_engine_type,
    year=red_engine_my,
    hp=red_hp,
    load_factor=load_factor,
    emission_standard=red_standard,
)

# surplus = calc_surplus_emissions(
#     red_equip=red,
#     base_equip=base,
#     project_life=project_life,
#     year_1=year_1,
#     load_factor=load_factor,
#     annual_activity=312,  # Minimum activity
#     percent_op=percent_op,
#     verbose=True,
# )
surplus = calc_surplus_emissions(
    baseline_engine=[base],
    reduced_engine=red,
    start_year=year_1,
    annual_activity=[312],
    percent_operation=[percent_op],
    project_life=project_life,
)

# min_act = min_annual_act(
#     red_equip=red,
#     base_equip=base,
#     year_1=year_1,
#     load_factor=load_factor,
#     annual_activity=annual_activity,
#     percent_op=percent_op,
#     ce_limit=30000,
#     cost_red_equip=92000,
#     max_percent=0.85,
#     rate=0.01,
#     project_life=5,
#     tol=1000,
#     step=1,
# )
min_act = min_annual_activity(
    baseline_engine=[base],
    reduced_engine=red,
    start_year=year_1,
    annual_activity=[annual_activity],
    percent_operation=[percent_op],
    project_life=5,
    ce_limit=30000,
    cost_reduced_engine=92000,
    max_percent=0.85,
    rate=0.01,
    iterations=20,
)

print(min_act)
