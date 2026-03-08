import streamlit as st
import pandas as pd
import numpy as np
import random
from collections import defaultdict
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pulp
import plotly.express as px
import plotly.graph_objects as go

nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("punkt_tab", quiet=True)

# =============================================================================
# 1. DIGITAL TWIN SIMULATOR
# =============================================================================
class DigitalTwinSimulator:
    def __init__(self, devices_df, environment_df, rules_df):
        self.devices = devices_df.to_dict("records")
        self.environment = environment_df.to_dict("records")
        self.rules = rules_df.to_dict("records")
        self.virtual_home = []
        self.total_energy = 0.0
        self.comfort_score = 0
        self.time_step_results = []

    def initialize_virtual_home(self):
        for d in self.devices:
            self.virtual_home.append({"id": d["device_id"], "state": d["state"],
                                      "power": d["power_rating"], "current_state": d["state"]})

    def apply_rule(self, rule, env):
        value = env.get(rule["condition_field"])
        if value is None:
            return
        op = rule["operator"]
        t = rule["threshold"]
        met = (op == ">" and value > t) or (op == "<" and value < t) or               (op == "==" and value == t) or (op == ">=" and value >= t) or (op == "<=" and value <= t)
        if met:
            for dev in self.virtual_home:
                if dev["id"] == rule["device_id"]:
                    dev["current_state"] = rule["action"]
                    break

    def calculate_energy(self, time_interval):
        return sum(dev["power"] * time_interval for dev in self.virtual_home if dev["current_state"] == "ON")

    def calculate_comfort(self, temperature):
        return 1 if 22 <= temperature <= 26 else -1

    def run_simulation(self, time_interval=1.0):
        self.initialize_virtual_home()
        for t, env in enumerate(self.environment):
            for rule in self.rules:
                self.apply_rule(rule, env)
            energy_step = self.calculate_energy(time_interval)
            self.total_energy += energy_step
            comfort_step = self.calculate_comfort(env["temperature"])
            self.comfort_score += comfort_step
            self.time_step_results.append({
                "time_step": t, "temperature": env["temperature"],
                "humidity": env["humidity"], "occupancy": env["occupancy"],
                "device_states": [dev["current_state"] for dev in self.virtual_home],
                "energy_step": energy_step, "cumulative_energy": self.total_energy,
                "cumulative_comfort": self.comfort_score
            })
        return pd.DataFrame(self.time_step_results)


# =============================================================================
# 2. CONFLICT DETECTION
# =============================================================================
class ConflictDetector:
    def __init__(self, rules_df):
        self.rules = rules_df.to_dict("records")
        self.graph = defaultdict(list)

    def build_graph(self):
        for i, rule in enumerate(self.rules):
            self.graph[rule["device_id"]].append(i)

    def detect_conflicts(self):
        conflicts = []
        for device, indices in self.graph.items():
            if len(indices) > 1:
                for idx in indices:
                    for jdx in indices:
                        if idx >= jdx:
                            continue
                        r1, r2 = self.rules[idx], self.rules[jdx]
                        if r1["condition_field"] == r2["condition_field"] and r1["action"] != r2["action"]:
                            conflicts.append((idx, jdx))
        return conflicts

    def resolve_conflict(self, conflict_pair):
        idx, jdx = conflict_pair
        return self.rules[idx] if self.rules[idx]["priority"] < self.rules[jdx]["priority"] else self.rules[jdx]

    def run(self):
        self.build_graph()
        conflicts = self.detect_conflicts()
        return [self.resolve_conflict(c) for c in conflicts], conflicts


# =============================================================================
# 3. LSTM ENERGY PREDICTOR (Pure NumPy - no TensorFlow)
# =============================================================================
class LSTMEnergyPredictor:
    """Single-layer LSTM implemented in NumPy. No TensorFlow dependency."""
    def __init__(self, energy_df, sequence_length=10):
        self.energy = energy_df["energy"].values.astype(float)
        self.seq_len = sequence_length
        self.scaler_mean = None
        self.scaler_std = None
        self.hidden_size = 32
        self._init_weights()

    def _init_weights(self):
        H, scale = self.hidden_size, 0.1
        rng = np.random.default_rng(42)
        self.Wf = rng.standard_normal((H, 1 + H)) * scale
        self.Wi = rng.standard_normal((H, 1 + H)) * scale
        self.Wc = rng.standard_normal((H, 1 + H)) * scale
        self.Wo = rng.standard_normal((H, 1 + H)) * scale
        self.bf = np.ones((H, 1))
        self.bi = np.zeros((H, 1))
        self.bc = np.zeros((H, 1))
        self.bo = np.zeros((H, 1))
        self.Wy = rng.standard_normal((1, H)) * scale
        self.by = np.zeros((1, 1))

    @staticmethod
    def _sigmoid(x):
        return 1 / (1 + np.exp(-np.clip(x, -15, 15)))

    def _forward_sequence(self, seq):
        H = self.hidden_size
        h = np.zeros((H, 1))
        c = np.zeros((H, 1))
        for t in range(len(seq)):
            x_t = seq[t].reshape(1, 1)
            xh = np.vstack([x_t, h])
            f = self._sigmoid(self.Wf @ xh + self.bf)
            i = self._sigmoid(self.Wi @ xh + self.bi)
            c_hat = np.tanh(self.Wc @ xh + self.bc)
            o = self._sigmoid(self.Wo @ xh + self.bo)
            c = f * c + i * c_hat
            h = o * np.tanh(c)
        y = self.Wy @ h + self.by
        return y[0, 0], h

    def preprocess(self):
        self.scaler_mean = self.energy.mean()
        self.scaler_std = self.energy.std() + 1e-8
        norm = (self.energy - self.scaler_mean) / self.scaler_std
        X, y = [], []
        for i in range(len(norm) - self.seq_len):
            X.append(norm[i:i+self.seq_len])
            y.append(norm[i+self.seq_len])
        return np.array(X), np.array(y)

    def train(self, epochs=30, lr=0.005, validation_split=0.2):
        X, y = self.preprocess()
        split = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]
        best_val_rmse, no_improve, patience = float("inf"), 0, 5
        for epoch in range(epochs):
            idxs = np.random.permutation(len(X_train))
            for idx in idxs:
                seq = X_train[idx].reshape(-1, 1)
                target = y_train[idx]
                pred, h = self._forward_sequence(seq)
                loss_grad = 2 * (pred - target)
                self.Wy -= lr * loss_grad * h.T
                self.by -= lr * np.array([[loss_grad]])
                dh = self.Wy.T * loss_grad
                xh_last = np.vstack([seq[-1].reshape(1,1), h])
                gs = lr * 0.01
                for W in [self.Wf, self.Wi, self.Wc, self.Wo]:
                    W -= gs * dh @ xh_last.T
            val_preds = np.array([self._forward_sequence(X_val[i].reshape(-1,1))[0] for i in range(len(X_val))])
            val_rmse = np.sqrt(np.mean((val_preds - y_val)**2))
            if val_rmse < best_val_rmse:
                best_val_rmse, no_improve = val_rmse, 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break
        return best_val_rmse * self.scaler_std

    def predict_next(self, last_sequence):
        norm = (np.array(last_sequence) - self.scaler_mean) / self.scaler_std
        pred_norm, _ = self._forward_sequence(norm.reshape(-1, 1))
        return pred_norm * self.scaler_std + self.scaler_mean


# =============================================================================
# 4. LP OPTIMIZER
# =============================================================================
class LPOptimizer:
    def __init__(self, devices_df, current_temp, occupancy, time_interval=1.0):
        self.devices = devices_df.to_dict("records")
        self.current_temp = current_temp
        self.occupancy = occupancy
        self.dt = time_interval
        self.outside_temp = 20.0

    def optimize(self):
        prob = pulp.LpProblem("Minimize_Energy", pulp.LpMinimize)
        dv = {d["device_id"]: pulp.LpVariable(d["device_id"], lowBound=0, upBound=1, cat="Binary") for d in self.devices}
        prob += pulp.lpSum([d["power_rating"] * self.dt * dv[d["device_id"]] for d in self.devices])
        heater_id = next((d["device_id"] for d in self.devices if d["device_type"] == "heater"), None)
        cooler_id = next((d["device_id"] for d in self.devices if d["device_type"] == "cooler"), None)
        temp_next = (self.current_temp + 2.0 * dv[heater_id] - 2.0 * dv[cooler_id]
                     - 0.1 * (self.current_temp - self.outside_temp)) if heater_id and cooler_id else self.current_temp
        prob += temp_next >= 22
        prob += temp_next <= 26
        if self.occupancy == 1:
            for d in self.devices:
                if d["device_type"] == "light":
                    prob += dv[d["device_id"]] == 1
        prob.solve(pulp.PULP_CBC_CMD(msg=False))
        return {d["device_id"]: "ON" if dv[d["device_id"]].varValue > 0.5 else "OFF" for d in self.devices}, pulp.value(prob.objective)


# =============================================================================
# 5. NLP COMPILER
# =============================================================================
class NLPCompiler:
    def __init__(self, devices_df):
        self.devices_df = devices_df
        self.stop_words = set(stopwords.words("english"))
        self.vectorizer = TfidfVectorizer(max_features=100)
        self.intent_classifier = None
        self.device_map = {"light": "light1", "ac": "cooler1", "heater": "heater1",
                           "cooler": "cooler1", "fan": None, "tv": None, "humidifier": None}
        self._train_intent_model()

    def _train_intent_model(self):
        texts = ["turn on the light when it gets dark", "switch on ac if temperature above 25",
                 "set heater to on at 6pm", "turn off fan if no one home",
                 "please activate the humidifier", "what is the weather", "hello world"]
        intents = ["set_device"]*5 + ["unknown"]*2
        X = self.vectorizer.fit_transform(texts)
        self.intent_classifier = LogisticRegression()
        self.intent_classifier.fit(X, intents)

    def preprocess(self, text):
        tokens = word_tokenize(text.lower())
        return " ".join(w for w in tokens if w.isalnum() and w not in self.stop_words)

    def detect_intent(self, text):
        return self.intent_classifier.predict(self.vectorizer.transform([self.preprocess(text)]))[0]

    def extract_entities(self, text):
        tokens = word_tokenize(text.lower())
        device, condition, value = None, None, None
        for i, tok in enumerate(tokens):
            if tok in self.device_map:
                device = self.device_map[tok]
            if tok in {"temperature", "humidity", "occupancy"}:
                condition = tok
            if tok.isdigit():
                value = int(tok)
        return device, condition, value

    def generate_rule(self, command):
        if self.detect_intent(command) != "set_device":
            return None
        device, condition, value = self.extract_entities(command)
        if device is None:
            return None
        condition = condition or "temperature"
        value = value or 22
        operator = ">" if "above" in command or "higher" in command else "<" if "below" in command or "lower" in command else "=="
        action = "ON" if "on" in command else "OFF" if "off" in command else "ON"
        return {"device_id": device, "condition_field": condition, "operator": operator,
                "threshold": value, "action": action, "priority": 5}

    def validate_rule(self, rule):
        if rule is None or rule["device_id"] not in self.devices_df["device_id"].values:
            return False
        return rule["condition_field"] in ["temperature", "humidity", "occupancy"]

    def compile(self, nl_df):
        rules = [self.generate_rule(row["command"]) for _, row in nl_df.iterrows()]
        return pd.DataFrame([r for r in rules if self.validate_rule(r)])


# =============================================================================
# 6. RL RESOLVER
# =============================================================================
class RLConflictResolver:
    def __init__(self, rules_df, devices_df, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.rules = rules_df.to_dict("records")
        self.devices = devices_df.to_dict("records")
        self.alpha, self.gamma, self.epsilon = alpha, gamma, epsilon
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.device_rules = defaultdict(list)
        for i, r in enumerate(self.rules):
            self.device_rules[r["device_id"]].append(i)

    def _temp_bin(self, t):
        return "low" if t < 22 else "comfortable" if t <= 26 else "high"

    def _simulate(self, device, rule_idx, temp):
        rule = self.rules[rule_idx]
        dev = next((d for d in self.devices if d["device_id"] == device), None)
        if not dev:
            return temp, -1
        new_temp = temp
        if dev["device_type"] == "heater" and rule["action"] == "ON":
            new_temp += 2.0
        elif dev["device_type"] == "cooler" and rule["action"] == "ON":
            new_temp -= 2.0
        new_temp -= 0.1 * (temp - 20)
        return new_temp, (1 if 22 <= new_temp <= 26 else -1)

    def train(self, episodes=500):
        for _ in range(episodes):
            device = random.choice(list(self.device_rules.keys()))
            temp = random.uniform(18, 30)
            state = (device, self._temp_bin(temp))
            possible = self.device_rules[device]
            if not possible:
                continue
            action = random.choice(possible) if random.random() < self.epsilon else max(possible, key=lambda a: self.q_table[state][a])
            next_temp, reward = self._simulate(device, action, temp)
            next_state = (device, self._temp_bin(next_temp))
            best_next = max((self.q_table[next_state][a] for a in possible), default=0)
            self.q_table[state][action] += self.alpha * (reward + self.gamma * best_next - self.q_table[state][action])

    def resolve(self, device, temp, occupancy):
        possible = self.device_rules[device]
        if not possible:
            return None
        state = (device, self._temp_bin(temp))
        return self.rules[max(possible, key=lambda a: self.q_table[state][a])]


# =============================================================================
# DATA
# =============================================================================
@st.cache_data
def generate_data():
    devices = pd.DataFrame([
        {"device_id": "light1", "state": "OFF", "power_rating": 10, "device_type": "light"},
        {"device_id": "heater1", "state": "OFF", "power_rating": 2000, "device_type": "heater"},
        {"device_id": "cooler1", "state": "OFF", "power_rating": 1500, "device_type": "cooler"},
    ])
    times = list(range(24))
    temps = [20 + 5 * np.sin(i/24 * 2*np.pi) + random.uniform(-1,1) for i in times]
    hum = [50 + 10 * random.random() for _ in times]
    occ = [1 if 8 <= i <= 22 else 0 for i in times]
    env = pd.DataFrame({"time": times, "temperature": temps, "humidity": hum, "occupancy": occ})
    rules = pd.DataFrame([
        {"rule_id": 1, "device_id": "cooler1", "condition_field": "temperature", "operator": ">", "threshold": 26, "action": "ON", "priority": 1},
        {"rule_id": 2, "device_id": "heater1", "condition_field": "temperature", "operator": "<", "threshold": 18, "action": "ON", "priority": 2},
        {"rule_id": 3, "device_id": "light1", "condition_field": "occupancy", "operator": "==", "threshold": 1, "action": "ON", "priority": 3},
        {"rule_id": 4, "device_id": "cooler1", "condition_field": "temperature", "operator": ">", "threshold": 26, "action": "OFF", "priority": 4},
    ])
    np.random.seed(42)
    energy_vals = np.cumsum(np.random.randn(100) * 0.5) + 10
    energy = pd.DataFrame({"timestamp": range(100), "energy": energy_vals})
    nl = pd.DataFrame([
        {"command": "turn on the ac if temperature above 25"},
        {"command": "switch off lights when no one home"},
        {"command": "set heater to on at 6pm"}
    ])
    return devices, env, rules, energy, nl

@st.cache_resource
def train_lstm(energy_df):
    pred = LSTMEnergyPredictor(energy_df, sequence_length=10)
    rmse = pred.train(epochs=30)
    return pred, rmse

@st.cache_resource
def train_rl(rules_df, devices_df):
    rl = RLConflictResolver(rules_df, devices_df)
    rl.train(episodes=500)
    return rl

# =============================================================================
# APP
# =============================================================================
# Page config handled by main app.py
st.title("🏠 Smart Home Digital Twin Dashboard")

st.sidebar.header("Simulation Controls")
time_interval = st.sidebar.slider("Time interval (hours)", 0.5, 2.0, 1.0, 0.1)
run_button = st.sidebar.button("Run Full Simulation")

devices_df, env_df, rules_df, energy_df, nl_df = generate_data()

if "lstm_predictor" not in st.session_state:
    with st.spinner("Training LSTM energy predictor (NumPy)..."):
        st.session_state.lstm_predictor, st.session_state.lstm_rmse = train_lstm(energy_df)
if "rl_resolver" not in st.session_state:
    with st.spinner("Training RL conflict resolver..."):
        st.session_state.rl_resolver = train_rl(rules_df, devices_df)

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Simulation", "Conflict Detection", "Energy Prediction",
    "Optimization", "NLP Compiler", "RL Resolver", "Summary"
])

with tab1:
    st.header("Digital Twin Simulation")
    if run_button:
        with st.spinner("Running simulation..."):
            sim = DigitalTwinSimulator(devices_df, env_df, rules_df)
            sim_results = sim.run_simulation(time_interval=time_interval)
        st.success("Simulation complete!")
        c1, c2 = st.columns(2)
        c1.metric("Total Energy (Wh)", f"{sim.total_energy:.2f}")
        c2.metric("Comfort Score", f"{sim.comfort_score}")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sim_results["time_step"], y=sim_results["temperature"], mode="lines+markers", name="Temperature"))
        fig.add_trace(go.Scatter(x=sim_results["time_step"], y=sim_results["energy_step"], mode="lines+markers", name="Energy Step", yaxis="y2"))
        fig.update_layout(title="Environmental and Energy Data", xaxis_title="Time Step", yaxis_title="Temperature (°C)",
                          yaxis2=dict(title="Energy (Wh)", overlaying="y", side="right"))
        st.plotly_chart(fig, use_container_width=True)
        device_names = [d["id"] for d in sim.virtual_home]
        states_matrix = [[1 if s == "ON" else 0 for s in r["device_states"]] for r in sim_results.to_dict("records")]
        fig2 = px.imshow(np.array(states_matrix).T, x=sim_results["time_step"], y=device_names,
                         color_continuous_scale="Viridis", title="Device States (ON=1, OFF=0)")
        st.plotly_chart(fig2, use_container_width=True)
        st.dataframe(sim_results)
    else:
        st.info("Click 'Run Full Simulation' in the sidebar to start.")

with tab2:
    st.header("Conflict Detection (Rule Dependency Graph)")
    detector = ConflictDetector(rules_df)
    resolved_rules, conflicts = detector.run()
    st.subheader("Detected Conflicts")
    if conflicts:
        for (i, j) in conflicts:
            r1, r2 = rules_df.iloc[i], rules_df.iloc[j]
            st.warning(f"Conflict between Rule {r1['rule_id']} and Rule {r2['rule_id']} on device {r1['device_id']}")
    else:
        st.success("No conflicts detected.")
    st.subheader("Resolved Rules (by priority)")
    if resolved_rules:
        st.dataframe(pd.DataFrame(resolved_rules))
    st.subheader("Rule Graph")
    st.dataframe(pd.DataFrame([{"device": d, "rule_indices": idxs} for d, idxs in detector.graph.items()]))

with tab3:
    st.header("LSTM Energy Prediction")
    st.caption("Pure NumPy LSTM — no TensorFlow required. Compatible with any Python version.")
    st.metric("Training RMSE (denormalized)", f"{st.session_state.lstm_rmse:.2f} Wh")
    next_pred = st.session_state.lstm_predictor.predict_next(energy_df["energy"].values[-10:])
    st.metric("Next Energy Prediction", f"{next_pred:.2f} Wh")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=energy_df["timestamp"], y=energy_df["energy"], mode="lines", name="Historical"))
    fig.add_trace(go.Scatter(x=[energy_df["timestamp"].iloc[-1]+1], y=[next_pred], mode="markers",
                              marker=dict(size=10, color="red"), name="Next Prediction"))
    fig.update_layout(title="Energy Consumption History and Next Prediction", xaxis_title="Time", yaxis_title="Energy (Wh)")
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.header("Linear Programming Optimization")
    last_env = env_df.iloc[-1]
    st.write(f"Current temperature: {last_env['temperature']:.1f}°C, Occupancy: {'Yes' if last_env['occupancy'] else 'No'}")
    opt = LPOptimizer(devices_df, last_env["temperature"], last_env["occupancy"], time_interval)
    opt_settings, min_energy = opt.optimize()
    st.metric("Minimum Possible Energy (Wh)", f"{min_energy:.2f}")
    st.subheader("Optimal Device Settings")
    for dev, state in opt_settings.items():
        st.write(f"**{dev}**: {state}")

with tab5:
    st.header("NLP Rule Compiler")
    st.subheader("Natural Language Commands")
    st.dataframe(nl_df)
    nlp = NLPCompiler(devices_df)
    new_rules = nlp.compile(nl_df)
    st.subheader("Generated Rules")
    if not new_rules.empty:
        st.dataframe(new_rules)
    else:
        st.info("No valid rules generated.")
    st.subheader("Intent Detection Example")
    example = st.text_input("Enter a command to test:", "turn on the heater when temperature below 20")
    if example:
        intent = nlp.detect_intent(example)
        st.write(f"Detected intent: **{intent}**")
        if intent == "set_device":
            st.write("Generated rule:", nlp.generate_rule(example))

with tab6:
    st.header("RL Conflict Resolver")
    st.write("Q-learning agent trained on simulated environment.")
    q_sample = [{"state": s, "action": a, "Q-value": v}
                for (s, actions) in list(st.session_state.rl_resolver.q_table.items())[:5]
                for a, v in actions.items()]
    if q_sample:
        st.dataframe(pd.DataFrame(q_sample))
    else:
        st.info("Q-table is empty.")
    st.subheader("Resolve Conflict for a Device")
    device_choice = st.selectbox("Select device", devices_df["device_id"].tolist())
    temp_input = st.slider("Current temperature", 15.0, 35.0, 22.0)
    occ_input = st.checkbox("Occupied")
    if st.button("Resolve"):
        best_rule = st.session_state.rl_resolver.resolve(device_choice, temp_input, occ_input)
        if best_rule:
            st.success("Optimal rule according to Q-learning:")
            st.json(best_rule)
        else:
            st.warning("No conflicting rules for this device or state.")

with tab7:
    st.header("System Summary")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("**Devices**"); st.dataframe(devices_df)
    with col2:
        st.write("**Environment (first 5)**"); st.dataframe(env_df.head())
    with col3:
        st.write("**Rules**"); st.dataframe(rules_df)
    st.info("Run the simulation in the Simulation tab to see energy and comfort metrics.")
