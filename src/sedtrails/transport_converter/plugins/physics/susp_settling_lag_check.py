# Particle state update example for SedTRAILS: suspended vs bedload with probabilistic transitions
import numpy as np
import matplotlib.pyplot as plt

class Particle:
    def __init__(self, position, state="static"):
        self.position = position  # 2D position (x, y) on the bed
        self.state = state  # "static", "bedload", or "suspended"

class SedimentModel:
    def __init__(self, particles, shear_stress_func, dt, tau_c, tau_d, lambda_s_max, lambda_e_max, w_s, depth, kappa=0.4, u_star_func=None, d50=0.001):
        self.particles = particles
        self.tau = shear_stress_func  # function: position, time -> tau
        self.dt = dt  # timestep [s]
        self.tau_c = tau_c  # critical entrainment threshold
        self.tau_d = tau_d  # low-shear threshold for deposition
        self.lambda_s_max = lambda_s_max  # max settling rate (suspended -> bedload)
        self.lambda_e_max = lambda_e_max  # max entrainment rate (bedload -> suspended)
        self.w_s = w_s  # settling velocity [m/s]
        self.depth = depth  # water column depth [m]
        self.kappa = kappa  # von Karman constant
        self.u_star_func = u_star_func or self._default_u_star  # shear velocity function
        self.d50 = d50  # median grain size [m]
        self.bedload_thickness = 0.05#4 * d50  # bedload layer thickness [m]
        
        # Three-state tracking (Rouse method)
        self.static_history = []
        self.bedload_history = []
        self.suspended_history = []
        
        # MacDonald method tracking
        self.static_history_mac = []
        self.bedload_history_mac = []
        self.suspended_history_mac = []
        
        # MacDonald with settling lag method tracking
        self.static_history_lag = []
        self.bedload_history_lag = []
        self.suspended_history_lag = []
        
        # Common tracking
        self.time = 0  # current time [s]
        self.time_history = []
        self.tau_history = []
        self.rouse_history = []
        self.z_s_history = []  # MacDonald height history
        self.z_p_history = []  # Particle height with settling lag
        
        # Rouse number thresholds
        self.rouse_bedload_threshold = 2.5  # Ro > 2.5: bedload transport
        self.rouse_suspended_threshold = 2.5  # Ro < 2.5: suspended transport

    def _default_u_star(self, tau):
        """Calculate shear velocity from bed shear stress assuming water density = 1000 kg/m³"""
        rho_water = 1000  # kg/m³
        return np.sqrt(tau / rho_water)
    
    def calculate_rouse_number(self, tau):
        """Calculate Rouse number: Ro = w_s / (κ * u*)"""
        u_star = self.u_star_func(tau)
        return self.w_s / (self.kappa * u_star)
    
    def determine_transport_mode(self, rouse_number):
        """Determine transport mode based on Rouse number"""
        if rouse_number > self.rouse_bedload_threshold:
            return "static"  # No transport
        elif rouse_number > self.rouse_suspended_threshold:
            return "bedload"  # Bedload transport
        else:
            return "suspended"  # Suspended transport
    
    def calculate_macdonald_height(self, tau):
        """Calculate height of centroid of suspended load using MacDonald et al. (2006) Eq. 27"""
        u_star = self.u_star_func(tau)
        
        # Avoid log of negative or zero values
        rouse_param = self.w_s / (self.kappa * u_star)
            
        log_arg = np.log(rouse_param) - 0.4
        tanh_arg = 1.2 * log_arg
        
        # Calculate MacDonald height
        z_s = self.depth * 0.0398 * (10 ** (-1.08 * np.tanh(tanh_arg)))
        return z_s
    
    def determine_transport_mode_macdonald(self, tau):
        """Determine transport mode based on MacDonald height and bedload thickness"""
        if tau < self.tau_d:
            return "static"
        
        z_s = self.calculate_macdonald_height(tau)
        
        if z_s <= self.bedload_thickness and tau >= self.tau_d:
            return "bedload"
        elif z_s > self.bedload_thickness and tau >= self.tau_d:
            return "suspended"
        else:
            return "static"
    
    def calculate_particle_height_with_lag(self, tau):
        """Calculate particle height with settling lag: z_p(t) = z_s(t) when rising, z_p(t-1) - w_s*dt when falling"""
        z_s_current = self.calculate_macdonald_height(tau)
        
        if len(self.z_p_history) == 0:
            # First timestep - no history available
            return z_s_current
        
        z_s_previous = self.z_s_history[-1]  # Previous MacDonald height
        z_p_previous = self.z_p_history[-1]  # Previous particle height
        
        if z_s_current >= z_s_previous:
            # Flow increasing or steady - particles follow MacDonald height
            z_p_current = z_s_current
        else:
            # Flow decreasing - particles settle at fall velocity
            z_p_current = z_p_previous - self.w_s * self.dt
            # Don't let particles settle below the current MacDonald height
            z_p_current = max(z_p_current, z_s_current)
        
        return z_p_current
    
    def determine_transport_mode_lag(self, tau, z_p):
        """Determine transport mode based on lagged particle height and bedload thickness"""
        if tau < self.tau_d:
            return "static"
        
        if z_p <= self.bedload_thickness and tau >= self.tau_d:
            return "bedload"
        elif z_p > self.bedload_thickness and tau >= self.tau_d:
            return "suspended"
        else:
            return "static"
    
    def update_particle_states(self):
        # Get current shear stress (assuming uniform across all particles for this example)
        tau_local = self.tau(self.particles[0].position, self.time)
        rouse_number = self.calculate_rouse_number(tau_local)
        z_s = self.calculate_macdonald_height(tau_local)
        
        # Calculate particle height with settling lag
        z_p = self.calculate_particle_height_with_lag(tau_local)
        
        # Determine target modes for all three methods
        target_mode_rouse = self.determine_transport_mode(rouse_number)
        target_mode_mac = self.determine_transport_mode_macdonald(tau_local)
        target_mode_lag = self.determine_transport_mode_lag(tau_local, z_p)
        
        # Counters for Rouse method
        static_count = 0
        bedload_count = 0
        suspended_count = 0
        
        # Counters for MacDonald method
        static_count_mac = 0
        bedload_count_mac = 0
        suspended_count_mac = 0
        
        # Counters for MacDonald with settling lag method
        static_count_lag = 0
        bedload_count_lag = 0
        suspended_count_lag = 0
        
        for p in self.particles:
            # Initialize MacDonald states if not present
            if not hasattr(p, 'state_mac'):
                p.state_mac = "static"
            if not hasattr(p, 'state_lag'):
                p.state_lag = "static"
            
            r1 = np.random.rand()
            r2 = np.random.rand()
            r3 = np.random.rand()
            
            # ===== ROUSE METHOD =====
            current_state = p.state
            
            # Probabilistic transitions based on current state and target mode
            if current_state == "static" and target_mode_rouse != "static":
                if target_mode_rouse == "bedload" and tau_local > self.tau_d:
                    # Entrainment from static to bedload
                    lambda_e = self.lambda_e_max * 0.5 * (tau_local / self.tau_d - 1) if tau_local > self.tau_d else 0
                    P_entrain = 1 - np.exp(-max(lambda_e, 0) * self.dt)
                    if r1 < P_entrain:
                        p.state = "bedload"
                elif target_mode_rouse == "suspended" and tau_local > self.tau_c:
                    # Direct entrainment from static to suspended (strong flow)
                    lambda_e = self.lambda_e_max * (tau_local / self.tau_c - 1)
                    P_entrain = 1 - np.exp(-max(lambda_e, 0) * self.dt)
                    if r1 < P_entrain:
                        p.state = "suspended"
                        
            elif current_state == "bedload":
                if target_mode_rouse == "suspended" and tau_local > self.tau_c:
                    # Transition from bedload to suspended
                    lambda_e = self.lambda_e_max * 0.8 * (tau_local / self.tau_c - 1)
                    P_entrain = 1 - np.exp(-max(lambda_e, 0) * self.dt)
                    if r1 < P_entrain:
                        p.state = "suspended"
                elif target_mode_rouse == "static" and tau_local < self.tau_d:
                    # Deposition from bedload to static
                    lambda_s = self.lambda_s_max * 1.5
                    P_settle = 1 - np.exp(-lambda_s * self.dt)
                    if r1 < P_settle:
                        p.state = "static"
                        
            elif current_state == "suspended":
                if target_mode_rouse == "bedload":
                    # Settling from suspended to bedload
                    if tau_local < self.tau_c and tau_local > self.tau_d:
                        lambda_s = self.lambda_s_max * (self.tau_c - tau_local) / (self.tau_c - self.tau_d)
                        P_settle = 1 - np.exp(-lambda_s * self.dt)
                        if r1 < P_settle:
                            p.state = "bedload"
                elif target_mode_rouse == "static" and tau_local < self.tau_d:
                    # Direct settling from suspended to static (low flow)
                    lambda_s = self.lambda_s_max * 2.0
                    P_settle = 1 - np.exp(-lambda_s * self.dt)
                    if r1 < P_settle:
                        p.state = "static"
            
            # ===== MACDONALD METHOD =====
            current_state_mac = p.state_mac
            
            # MacDonald method: simpler transitions based on height threshold
            if current_state_mac == "static" and tau_local >= self.tau_c:
                # Only transition from static if tau exceeds tau_c
                lambda_e = self.lambda_e_max * (tau_local / self.tau_c - 1)
                P_entrain = 1 - np.exp(-max(lambda_e, 0) * self.dt)
                if r2 < P_entrain:
                    p.state_mac = target_mode_mac
                    
            elif current_state_mac in ["bedload", "suspended"]:
                if tau_local < self.tau_d:
                    # Transition to static if below tau_d
                    lambda_s = self.lambda_s_max * 2.0
                    P_settle = 1 - np.exp(-lambda_s * self.dt)
                    if r2 < P_settle:
                        p.state_mac = "static"
                else:
                    # Stay mobile, but adjust between bedload/suspended based on height
                    if target_mode_mac != current_state_mac and target_mode_mac != "static":
                        # Transition between bedload and suspended
                        transition_rate = self.lambda_e_max * 0.5
                        P_transition = 1 - np.exp(-transition_rate * self.dt)
                        if r2 < P_transition:
                            p.state_mac = target_mode_mac
            
            # ===== MACDONALD WITH SETTLING LAG METHOD =====
            current_state_lag = p.state_lag
            
            # Settling lag method: same transitions but based on lagged particle height
            if current_state_lag == "static" and tau_local >= self.tau_c:
                # Only transition from static if tau exceeds tau_c
                lambda_e = self.lambda_e_max * (tau_local / self.tau_c - 1)
                P_entrain = 1 - np.exp(-max(lambda_e, 0) * self.dt)
                if r3 < P_entrain:
                    p.state_lag = target_mode_lag
                    
            elif current_state_lag in ["bedload", "suspended"]:
                if tau_local < self.tau_d:
                    # Transition to static if below tau_d
                    lambda_s = self.lambda_s_max * 2.0
                    P_settle = 1 - np.exp(-lambda_s * self.dt)
                    if r3 < P_settle:
                        p.state_lag = "static"
                else:
                    # Stay mobile, but adjust between bedload/suspended based on lagged height
                    if target_mode_lag != current_state_lag and target_mode_lag != "static":
                        # Transition between bedload and suspended
                        transition_rate = self.lambda_e_max * 0.5
                        P_transition = 1 - np.exp(-transition_rate * self.dt)
                        if r3 < P_transition:
                            p.state_lag = target_mode_lag
            
            # Count particles in each state for both methods
            # Rouse method
            if p.state == "static":
                static_count += 1
            elif p.state == "bedload":
                bedload_count += 1
            elif p.state == "suspended":
                suspended_count += 1
            
            # MacDonald method
            if p.state_mac == "static":
                static_count_mac += 1
            elif p.state_mac == "bedload":
                bedload_count_mac += 1
            elif p.state_mac == "suspended":
                suspended_count_mac += 1
            
            # MacDonald with settling lag method
            if p.state_lag == "static":
                static_count_lag += 1
            elif p.state_lag == "bedload":
                bedload_count_lag += 1
            elif p.state_lag == "suspended":
                suspended_count_lag += 1
        
        # Store state fractions for both methods
        total_particles = len(self.particles)
        
        # Rouse method
        self.static_history.append(static_count / total_particles)
        self.bedload_history.append(bedload_count / total_particles)
        self.suspended_history.append(suspended_count / total_particles)
        
        # MacDonald method
        self.static_history_mac.append(static_count_mac / total_particles)
        self.bedload_history_mac.append(bedload_count_mac / total_particles)
        self.suspended_history_mac.append(suspended_count_mac / total_particles)
        
        # MacDonald with settling lag method
        self.static_history_lag.append(static_count_lag / total_particles)
        self.bedload_history_lag.append(bedload_count_lag / total_particles)
        self.suspended_history_lag.append(suspended_count_lag / total_particles)
        
        # Common tracking
        self.time_history.append(self.time / 3600)  # convert to hours
        self.tau_history.append(tau_local)
        self.rouse_history.append(rouse_number)
        self.z_s_history.append(z_s)
        self.z_p_history.append(z_p)
        self.time += self.dt

    def find_threshold_crossings(self):
        """Find where bed shear stress crosses tau_c and tau_d thresholds"""
        tau_c_crossings = []
        tau_d_crossings = []
        
        if len(self.tau_history) < 2:
            return tau_c_crossings, tau_d_crossings
        
        for i in range(1, len(self.tau_history)):
            tau_prev = self.tau_history[i-1]
            tau_curr = self.tau_history[i]
            time_prev = self.time_history[i-1]
            time_curr = self.time_history[i]
            
            # Check tau_c crossings
            if (tau_prev < self.tau_c and tau_curr >= self.tau_c) or (tau_prev >= self.tau_c and tau_curr < self.tau_c):
                # Linear interpolation to find exact crossing time
                if tau_curr != tau_prev:  # Avoid division by zero
                    crossing_time = time_prev + (self.tau_c - tau_prev) * (time_curr - time_prev) / (tau_curr - tau_prev)
                    tau_c_crossings.append(crossing_time)
            
            # Check tau_d crossings
            if (tau_prev < self.tau_d and tau_curr >= self.tau_d) or (tau_prev >= self.tau_d and tau_curr < self.tau_d):
                # Linear interpolation to find exact crossing time
                if tau_curr != tau_prev:  # Avoid division by zero
                    crossing_time = time_prev + (self.tau_d - tau_prev) * (time_curr - time_prev) / (tau_curr - tau_prev)
                    tau_d_crossings.append(crossing_time)
        
        return tau_c_crossings, tau_d_crossings

    def advect_particles(self, flow_velocity_func):
        for p in self.particles:
            u = flow_velocity_func(p.position)
            if p.state == "suspended":
                # Suspended particles move at full flow velocity
                p.position += u * self.dt
            elif p.state == "bedload":
                # Bedload particles move slower (near-bed transport)
                p.position += 0.3 * u * self.dt
            # Static particles don't move

    def plot_particle_distribution(self):
        x_static = [p.position[0] for p in self.particles if p.state == "static"]
        x_bedload = [p.position[0] for p in self.particles if p.state == "bedload"]
        x_suspended = [p.position[0] for p in self.particles if p.state == "suspended"]
        y_static = [p.position[1] for p in self.particles if p.state == "static"]
        y_bedload = [p.position[1] for p in self.particles if p.state == "bedload"]
        y_suspended = [p.position[1] for p in self.particles if p.state == "suspended"]

        plt.figure(figsize=(12, 5))
        plt.scatter(x_static, y_static, c='gray', label='Static', alpha=0.6, s=20)
        plt.scatter(x_bedload, y_bedload, c='brown', label='Bedload', alpha=0.7, s=30)
        plt.scatter(x_suspended, y_suspended, c='blue', label='Suspended', alpha=0.8, s=40)
        plt.xlabel("x position (m)")
        plt.ylabel("y position (m)")
        plt.title("Particle Positions by Transport State")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_state_history(self):
        plt.figure(figsize=(14, 12))
        
        # Get threshold crossings
        tau_c_crossings, tau_d_crossings = self.find_threshold_crossings()
        
        # Plot 1: Shear stress and Rouse number over time
        plt.subplot(4, 1, 1)
        ax1 = plt.gca()
        ax1.plot(self.time_history, self.tau_history, 'k-', linewidth=2, label='Bed shear stress')
        ax1.axhline(y=self.tau_c, color='r', linestyle='--', alpha=0.7, label=f'τ_c = {self.tau_c}')
        ax1.axhline(y=self.tau_d, color='b', linestyle='--', alpha=0.7, label=f'τ_d = {self.tau_d}')
        
        # Add threshold crossing vertical lines
        for crossing_time in tau_c_crossings:
            ax1.axvline(x=crossing_time, color='r', linestyle=':', alpha=0.6, linewidth=1)
        for crossing_time in tau_d_crossings:
            ax1.axvline(x=crossing_time, color='b', linestyle=':', alpha=0.6, linewidth=1)
        
        ax1.set_ylabel("Shear stress (Pa)", color='k')
        ax1.tick_params(axis='y', labelcolor='k')
        
        ax2 = ax1.twinx()
        ax2.plot(self.time_history, self.rouse_history, 'purple', linewidth=2, label='Rouse number')
        ax2.axhline(y=self.rouse_bedload_threshold, color='orange', linestyle=':', alpha=0.7, label=f'Ro_bedload = {self.rouse_bedload_threshold}')
        ax2.axhline(y=self.rouse_suspended_threshold, color='cyan', linestyle=':', alpha=0.7, label=f'Ro_suspended = {self.rouse_suspended_threshold}')
        ax2.set_ylabel("Rouse number", color='purple')
        ax2.tick_params(axis='y', labelcolor='purple')
        
        # Separate legends for clarity
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')
        plt.title("Tidal Forcing and Rouse Number")
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Three-state particle fractions (stacked)
        plt.subplot(4, 1, 2)
        static_arr = np.array(self.static_history)
        bedload_arr = np.array(self.bedload_history)
        suspended_arr = np.array(self.suspended_history)
        
        plt.fill_between(self.time_history, 0, static_arr, alpha=0.7, color='gray', label='Static')
        plt.fill_between(self.time_history, static_arr, static_arr + bedload_arr, alpha=0.7, color='brown', label='Bedload')
        plt.fill_between(self.time_history, static_arr + bedload_arr, 1, alpha=0.7, color='blue', label='Suspended')
        
        # Add threshold crossing vertical lines
        for crossing_time in tau_c_crossings:
            plt.axvline(x=crossing_time, color='r', linestyle=':', alpha=0.6, linewidth=1)
        for crossing_time in tau_d_crossings:
            plt.axvline(x=crossing_time, color='b', linestyle=':', alpha=0.6, linewidth=1)
        
        plt.ylabel("Particle Fraction")
        plt.title("Three-State Particle Distribution")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        
        # Plot 3: Individual state fractions
        plt.subplot(4, 1, 3)
        plt.plot(self.time_history, static_arr, 'gray', linewidth=2, label='Static', alpha=0.8)
        plt.plot(self.time_history, bedload_arr, 'brown', linewidth=2, label='Bedload', alpha=0.8)
        plt.plot(self.time_history, suspended_arr, 'blue', linewidth=2, label='Suspended', alpha=0.8)
        
        # Add threshold crossing vertical lines
        for crossing_time in tau_c_crossings:
            plt.axvline(x=crossing_time, color='r', linestyle=':', alpha=0.6, linewidth=1)
        for crossing_time in tau_d_crossings:
            plt.axvline(x=crossing_time, color='b', linestyle=':', alpha=0.6, linewidth=1)
        
        plt.ylabel("Particle Fraction")
        plt.title("Individual State Fractions Over Time")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        
        # Plot 4: Mobility classification
        plt.subplot(4, 1, 4)
        stationary_fraction = static_arr
        
        plt.fill_between(self.time_history, 0, stationary_fraction, alpha=0.6, color='gray', label='Stationary (Static)')
        plt.fill_between(self.time_history, stationary_fraction, 1, alpha=0.6, color='green', label='Moving (Bedload + Suspended)')
        
        # Add threshold crossing vertical lines
        for crossing_time in tau_c_crossings:
            plt.axvline(x=crossing_time, color='r', linestyle=':', alpha=0.6, linewidth=1)
        for crossing_time in tau_d_crossings:
            plt.axvline(x=crossing_time, color='b', linestyle=':', alpha=0.6, linewidth=1)
        
        plt.xlabel("Time (hours)")
        plt.ylabel("Particle Fraction")
        plt.title("Particle Mobility: Stationary vs Moving")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        
        plt.tight_layout()
        plt.show()
    
    def plot_method_comparison(self):
        """Plot comparison between Rouse and MacDonald methods"""
        plt.figure(figsize=(16, 10))
        
        # Get threshold crossings
        tau_c_crossings, tau_d_crossings = self.find_threshold_crossings()
        
        # Plot 1: Forcing conditions
        plt.subplot(3, 3, 1)
        plt.plot(self.time_history, self.tau_history, 'k-', linewidth=2, label='Bed shear stress')
        plt.axhline(y=self.tau_c, color='r', linestyle='--', alpha=0.7, label=f'τ_c = {self.tau_c}')
        plt.axhline(y=self.tau_d, color='b', linestyle='--', alpha=0.7, label=f'τ_d = {self.tau_d}')
        
        # Add threshold crossing vertical lines
        for crossing_time in tau_c_crossings:
            plt.axvline(x=crossing_time, color='r', linestyle=':', alpha=0.6, linewidth=1)
        for crossing_time in tau_d_crossings:
            plt.axvline(x=crossing_time, color='b', linestyle=':', alpha=0.6, linewidth=1)
        
        plt.ylabel("Shear stress (Pa)")
        plt.title("Tidal Forcing")
        plt.legend(fontsize=8)
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Classification parameters
        plt.subplot(3, 3, 2)
        ax1 = plt.gca()
        ax1.plot(self.time_history, self.rouse_history, 'purple', linewidth=2, label='Rouse number')
        ax1.axhline(y=self.rouse_bedload_threshold, color='orange', linestyle=':', alpha=0.7, label=f'Ro_bl = {self.rouse_bedload_threshold}')
        ax1.axhline(y=self.rouse_suspended_threshold, color='cyan', linestyle=':', alpha=0.7, label=f'Ro_susp = {self.rouse_suspended_threshold}')
        
        # Add threshold crossing vertical lines
        for crossing_time in tau_c_crossings:
            ax1.axvline(x=crossing_time, color='r', linestyle=':', alpha=0.6, linewidth=1)
        for crossing_time in tau_d_crossings:
            ax1.axvline(x=crossing_time, color='b', linestyle=':', alpha=0.6, linewidth=1)
        
        ax1.set_ylabel("Rouse number", color='purple')
        ax1.tick_params(axis='y', labelcolor='purple')
        
        ax2 = ax1.twinx()
        ax2.plot(self.time_history, np.array(self.z_s_history) * 1000, 'green', linewidth=2, label='MacDonald height z_s')
        ax2.plot(self.time_history, np.array(self.z_p_history) * 1000, 'orange', linewidth=2, label='Lagged height z_p', linestyle='--')
        ax2.axhline(y=self.bedload_thickness * 1000, color='brown', linestyle='-.', alpha=0.7, label='BL thickness')
        ax2.set_ylabel("Height (mm)", color='green')
        ax2.set_yscale('log')  # Set logarithmic scale for MacDonald height
        ax2.grid(True, which="both", ls="-", alpha=0.2)  # Add logarithmic grid
        ax2.tick_params(axis='y', labelcolor='green')
        
        ax1.legend(loc='upper left', fontsize=8)
        ax2.legend(loc='upper right', fontsize=8)
        plt.title("Classification Parameters")
        
        # Plot 3: Rouse Method - Stacked
        plt.subplot(3, 3, 4)
        static_arr = np.array(self.static_history)
        bedload_arr = np.array(self.bedload_history)
        suspended_arr = np.array(self.suspended_history)
        
        plt.fill_between(self.time_history, 0, static_arr, alpha=0.7, color='gray', label='Static')
        plt.fill_between(self.time_history, static_arr, static_arr + bedload_arr, alpha=0.7, color='brown', label='Bedload')
        plt.fill_between(self.time_history, static_arr + bedload_arr, 1, alpha=0.7, color='blue', label='Suspended')
        
        # Add threshold crossing vertical lines
        for crossing_time in tau_c_crossings:
            plt.axvline(x=crossing_time, color='r', linestyle=':', alpha=0.6, linewidth=1)
        for crossing_time in tau_d_crossings:
            plt.axvline(x=crossing_time, color='b', linestyle=':', alpha=0.6, linewidth=1)
        
        plt.ylabel("Particle Fraction")
        plt.title("Rouse Method - Stacked")
        plt.legend(fontsize=8)
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        
        # Plot 4: MacDonald Method - Stacked
        plt.subplot(3, 3, 5)
        static_arr_mac = np.array(self.static_history_mac)
        bedload_arr_mac = np.array(self.bedload_history_mac)
        suspended_arr_mac = np.array(self.suspended_history_mac)
        
        plt.fill_between(self.time_history, 0, static_arr_mac, alpha=0.7, color='gray', label='Static')
        plt.fill_between(self.time_history, static_arr_mac, static_arr_mac + bedload_arr_mac, alpha=0.7, color='brown', label='Bedload')
        plt.fill_between(self.time_history, static_arr_mac + bedload_arr_mac, 1, alpha=0.7, color='blue', label='Suspended')
        
        # Add threshold crossing vertical lines
        for crossing_time in tau_c_crossings:
            plt.axvline(x=crossing_time, color='r', linestyle=':', alpha=0.6, linewidth=1)
        for crossing_time in tau_d_crossings:
            plt.axvline(x=crossing_time, color='b', linestyle=':', alpha=0.6, linewidth=1)
        
        plt.ylabel("Particle Fraction")
        plt.title("MacDonald Method - Stacked")
        plt.legend(fontsize=8)
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        
        # Plot 5: MacDonald with Settling Lag - Stacked
        plt.subplot(3, 3, 6)
        static_arr_lag = np.array(self.static_history_lag)
        bedload_arr_lag = np.array(self.bedload_history_lag)
        suspended_arr_lag = np.array(self.suspended_history_lag)
        
        plt.fill_between(self.time_history, 0, static_arr_lag, alpha=0.7, color='gray', label='Static')
        plt.fill_between(self.time_history, static_arr_lag, static_arr_lag + bedload_arr_lag, alpha=0.7, color='brown', label='Bedload')
        plt.fill_between(self.time_history, static_arr_lag + bedload_arr_lag, 1, alpha=0.7, color='blue', label='Suspended')
        
        # Add threshold crossing vertical lines
        for crossing_time in tau_c_crossings:
            plt.axvline(x=crossing_time, color='r', linestyle=':', alpha=0.6, linewidth=1)
        for crossing_time in tau_d_crossings:
            plt.axvline(x=crossing_time, color='b', linestyle=':', alpha=0.6, linewidth=1)
        
        plt.ylabel("Particle Fraction")
        plt.title("MacDonald + Lag - Stacked")
        plt.legend(fontsize=8)
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        
        # Plot 6: Rouse Method Individual States
        plt.subplot(3, 3, 7)
        plt.plot(self.time_history, static_arr, 'gray', linewidth=2, label='Static', alpha=0.8)
        plt.plot(self.time_history, bedload_arr, 'brown', linewidth=2, label='Bedload', alpha=0.8)
        plt.plot(self.time_history, suspended_arr, 'blue', linewidth=2, label='Suspended', alpha=0.8)
        
        # Add threshold crossing vertical lines
        for crossing_time in tau_c_crossings:
            plt.axvline(x=crossing_time, color='r', linestyle=':', alpha=0.6, linewidth=1)
        for crossing_time in tau_d_crossings:
            plt.axvline(x=crossing_time, color='b', linestyle=':', alpha=0.6, linewidth=1)
        
        plt.xlabel("Time (hours)")
        plt.ylabel("Particle Fraction")
        plt.title("Rouse - Individual States")
        plt.legend(fontsize=8)
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        
        # Plot 7: MacDonald Method Individual States
        plt.subplot(3, 3, 8)
        plt.plot(self.time_history, static_arr_mac, 'gray', linewidth=2, label='Static', alpha=0.8, linestyle='--')
        plt.plot(self.time_history, bedload_arr_mac, 'brown', linewidth=2, label='Bedload', alpha=0.8, linestyle='--')
        plt.plot(self.time_history, suspended_arr_mac, 'blue', linewidth=2, label='Suspended', alpha=0.8, linestyle='--')
        
        # Add threshold crossing vertical lines
        for crossing_time in tau_c_crossings:
            plt.axvline(x=crossing_time, color='r', linestyle=':', alpha=0.6, linewidth=1)
        for crossing_time in tau_d_crossings:
            plt.axvline(x=crossing_time, color='b', linestyle=':', alpha=0.6, linewidth=1)
        
        plt.xlabel("Time (hours)")
        plt.ylabel("Particle Fraction")
        plt.title("MacDonald - Individual States")
        plt.legend(fontsize=8)
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        
        # Plot 8: MacDonald + Lag Method Individual States
        plt.subplot(3, 3, 9)
        plt.plot(self.time_history, static_arr_lag, 'gray', linewidth=2, label='Static', alpha=0.8, linestyle=':')
        plt.plot(self.time_history, bedload_arr_lag, 'brown', linewidth=2, label='Bedload', alpha=0.8, linestyle=':')
        plt.plot(self.time_history, suspended_arr_lag, 'blue', linewidth=2, label='Suspended', alpha=0.8, linestyle=':')
        
        # Add threshold crossing vertical lines
        for crossing_time in tau_c_crossings:
            plt.axvline(x=crossing_time, color='r', linestyle=':', alpha=0.6, linewidth=1)
        for crossing_time in tau_d_crossings:
            plt.axvline(x=crossing_time, color='b', linestyle=':', alpha=0.6, linewidth=1)
        
        plt.xlabel("Time (hours)")
        plt.ylabel("Particle Fraction")
        plt.title("MacDonald + Lag - Individual States")
        plt.legend(fontsize=8)
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        
        # Plot 8: Mobile Particles Comparison
        plt.subplot(3, 3, 9)
        mobile_rouse = bedload_arr + suspended_arr
        mobile_mac = bedload_arr_mac + suspended_arr_mac
        
        plt.plot(self.time_history, mobile_rouse, 'g-', linewidth=2, label='Rouse Method', alpha=0.8)
        plt.plot(self.time_history, mobile_mac, 'g--', linewidth=2, label='MacDonald Method', alpha=0.8)
        
        # Add threshold crossing vertical lines
        for crossing_time in tau_c_crossings:
            plt.axvline(x=crossing_time, color='r', linestyle=':', alpha=0.6, linewidth=1)
        for crossing_time in tau_d_crossings:
            plt.axvline(x=crossing_time, color='b', linestyle=':', alpha=0.6, linewidth=1)
        
        plt.xlabel("Time (hours)")
        plt.ylabel("Mobile Fraction")
        plt.title("Mobile Particles")
        plt.legend(fontsize=8)
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)

        plt.tight_layout()
        plt.show()
        
        # Create a dedicated comparison figure
        plt.figure(figsize=(12, 8))
        
        # Plot 1: Method Comparison - Suspended Particles
        plt.subplot(2, 2, 1)
        plt.plot(self.time_history, suspended_arr, 'b-', linewidth=2, label='Rouse Method', alpha=0.8)
        plt.plot(self.time_history, suspended_arr_mac, 'g--', linewidth=2, label='MacDonald Method', alpha=0.8)
        plt.plot(self.time_history, suspended_arr_lag, 'm:', linewidth=2, label='MacDonald + Lag', alpha=0.8)
        
        # Add threshold crossing vertical lines
        for crossing_time in tau_c_crossings:
            plt.axvline(x=crossing_time, color='r', linestyle=':', alpha=0.6, linewidth=1)
        for crossing_time in tau_d_crossings:
            plt.axvline(x=crossing_time, color='b', linestyle=':', alpha=0.6, linewidth=1)
        
        plt.ylabel("Suspended Fraction")
        plt.title("Suspended Particles - Method Comparison")
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        
        # Plot 2: Method Comparison - Mobile Particles (Bedload + Suspended)
        plt.subplot(2, 2, 2)
        mobile_rouse = bedload_arr + suspended_arr
        mobile_mac = bedload_arr_mac + suspended_arr_mac
        mobile_lag = bedload_arr_lag + suspended_arr_lag
        
        plt.plot(self.time_history, mobile_rouse, 'b-', linewidth=2, label='Rouse Method', alpha=0.8)
        plt.plot(self.time_history, mobile_mac, 'g--', linewidth=2, label='MacDonald Method', alpha=0.8)
        plt.plot(self.time_history, mobile_lag, 'm:', linewidth=2, label='MacDonald + Lag', alpha=0.8)
        
        # Add threshold crossing vertical lines
        for crossing_time in tau_c_crossings:
            plt.axvline(x=crossing_time, color='r', linestyle=':', alpha=0.6, linewidth=1)
        for crossing_time in tau_d_crossings:
            plt.axvline(x=crossing_time, color='b', linestyle=':', alpha=0.6, linewidth=1)
        
        plt.ylabel("Mobile Fraction")
        plt.title("Mobile Particles - Method Comparison")
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        
        # Plot 3: Height Comparison (z_s vs z_p)
        plt.subplot(2, 2, 3)
        plt.plot(self.time_history, self.z_s_history, 'purple', linewidth=2, label='MacDonald z_s', alpha=0.8)
        plt.plot(self.time_history, self.z_p_history, 'magenta', linewidth=2, linestyle='--', label='Lag z_p', alpha=0.8)
        plt.axhline(y=self.bedload_thickness, color='orange', linewidth=2, label='Bedload threshold', alpha=0.8)
        
        # Add threshold crossing vertical lines
        for crossing_time in tau_c_crossings:
            plt.axvline(x=crossing_time, color='r', linestyle=':', alpha=0.6, linewidth=1)
        for crossing_time in tau_d_crossings:
            plt.axvline(x=crossing_time, color='b', linestyle=':', alpha=0.6, linewidth=1)
        
        plt.ylabel("Height (m)")
        plt.title("Suspension Heights - z_s vs z_p")
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        # Plot 4: Hysteresis Effect (Settling Lag vs MacDonald)
        plt.subplot(2, 2, 4)
        # Calculate differences between methods
        diff_mac_rouse = suspended_arr_mac - suspended_arr
        diff_lag_mac = suspended_arr_lag - suspended_arr_mac
        diff_lag_rouse = suspended_arr_lag - suspended_arr
        
        plt.plot(self.time_history, diff_mac_rouse, 'g-', linewidth=2, label='MacDonald - Rouse', alpha=0.8)
        plt.plot(self.time_history, diff_lag_mac, 'm-', linewidth=2, label='Lag - MacDonald', alpha=0.8)
        plt.plot(self.time_history, diff_lag_rouse, 'k:', linewidth=2, label='Lag - Rouse', alpha=0.8)
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Add threshold crossing vertical lines
        for crossing_time in tau_c_crossings:
            plt.axvline(x=crossing_time, color='r', linestyle=':', alpha=0.6, linewidth=1)
        for crossing_time in tau_d_crossings:
            plt.axvline(x=crossing_time, color='b', linestyle=':', alpha=0.6, linewidth=1)
        
        plt.xlabel("Time (hours)")
        plt.ylabel("Suspension Difference")
        plt.title("Method Differences (Hysteresis Effects)")
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()# Example shear stress and flow velocity functions
def tidal_shear_stress(position, time):
    """12-hour sinusoidal tidal shear stress signal"""
    # 12 hours = 12 * 3600 = 43200 seconds period
    tidal_period = 12 * 3600  # 12 hours in seconds
    # Base shear stress + tidal variation
    base_tau = 0.0001  # average between tau_c and tau_d
    tidal_amplitude = 0.2  # amplitude to create variation around thresholds
    overtide_amplitude = 0.1  # overtide component
    overtide_period = 6 * 3600  # 6 hours in seconds
    return base_tau + np.abs(tidal_amplitude * np.sin(2 * np.pi * time / tidal_period) \
            + overtide_amplitude * np.sin(2 * np.pi * time / overtide_period - 3 * np.pi / 4))

def example_flow_velocity(position):
    return np.array([0.5, 0.0])

# Example setup with tidal forcing and three-state system
particles = [Particle(np.array([x, 0.0]), state="static") for x in np.linspace(0, 100, 200)]

# Create separate MacDonald-only model for individual testing
class MacDonaldOnlyModel(SedimentModel):
    """MacDonald method only for individual testing"""
    
    def update_particle_states(self):
        """MacDonald method only"""
        tau_local = self.tau(self.particles[0].position, self.time)
        z_s = self.calculate_macdonald_height(tau_local)
        target_mode = self.determine_transport_mode_macdonald(tau_local)
        
        static_count = 0
        bedload_count = 0
        suspended_count = 0
        
        for p in self.particles:
            if not hasattr(p, 'state_mac_only'):
                p.state_mac_only = "static"
            
            r = np.random.rand()
            current_state = p.state_mac_only
            
            # MacDonald method transitions
            if current_state == "static" and tau_local >= self.tau_c:
                lambda_e = self.lambda_e_max * (tau_local / self.tau_c - 1)
                P_entrain = 1 - np.exp(-max(lambda_e, 0) * self.dt)
                if r < P_entrain:
                    p.state_mac_only = target_mode
            elif current_state in ["bedload", "suspended"]:
                if tau_local < self.tau_d:
                    lambda_s = self.lambda_s_max * 2.0
                    P_settle = 1 - np.exp(-lambda_s * self.dt)
                    if r < P_settle:
                        p.state_mac_only = "static"
                else:
                    if target_mode != current_state and target_mode != "static":
                        transition_rate = self.lambda_e_max * 0.5
                        P_transition = 1 - np.exp(-transition_rate * self.dt)
                        if r < P_transition:
                            p.state_mac_only = target_mode
            
            # Count particles
            if p.state_mac_only == "static":
                static_count += 1
            elif p.state_mac_only == "bedload":
                bedload_count += 1
            elif p.state_mac_only == "suspended":
                suspended_count += 1
        
        # Store results
        total_particles = len(self.particles)
        self.static_history.append(static_count / total_particles)
        self.bedload_history.append(bedload_count / total_particles)
        self.suspended_history.append(suspended_count / total_particles)
        self.time_history.append(self.time / 3600)
        self.tau_history.append(tau_local)
        self.z_s_history.append(z_s)
        self.time += self.dt

# To test MacDonald method individually, uncomment the following:
# print("\n" + "="*60)
# print("TESTING MACDONALD METHOD INDIVIDUALLY")
# print("="*60)
# particles_mac = [Particle(np.array([x, 0.0]), state="static") for x in np.linspace(0, 100, 200)]
# model_mac = MacDonaldOnlyModel(
#     particles_mac, tidal_shear_stress, dt=300, tau_c=0.25, tau_d=0.05,
#     lambda_s_max=0.001, lambda_e_max=0.005, w_s=5e-3, depth=10.0, d50=0.001
# )
# 
# for step in range(total_steps):
#     model_mac.update_particle_states()
#     model_mac.advect_particles(example_flow_velocity)
# 
# print("MacDonald method only results:")
# model_mac.plot_state_history()
model = SedimentModel(
    particles,
    shear_stress_func=tidal_shear_stress,
    dt=60,  # 1-minute timestep
    tau_c=0.2,  # critical entrainment threshold
    tau_d=0.01,  # low-shear threshold for deposition
    lambda_s_max=0.001,
    lambda_e_max=0.005,
    w_s=5e-4,
    depth=10.0,  # 3m water depth for MacDonald calculation
    d50=0.0002  # 0.2mm median grain size
)

# Time loop - run for 24 hours (2 tidal cycles)
total_hours = 12
total_steps = int(total_hours * 3600 / model.dt)
print(f"Running simulation for {total_hours} hours ({total_steps} timesteps)")

for step in range(total_steps):
    model.update_particle_states()
    model.advect_particles(example_flow_velocity)
    if step % 144 == 0:  # Print progress every 12 hours
        print(f"Progress: {step/total_steps*100:.1f}% complete")

# Diagnostics
# print("\nShowing Rouse method results:")
# model.plot_state_history()

print("\nShowing method comparison:")
model.plot_method_comparison()

print("\nSummary:")
print(f"Bedload layer thickness: {model.bedload_thickness*1000:.1f} mm (4 × d50)")
print(f"Final MacDonald height range: {min(model.z_s_history)*1000:.1f} - {max(model.z_s_history)*1000:.1f} mm")
print(f"Water depth: {model.depth} m")