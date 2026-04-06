"""
Secure Swarm Communication Module
Implements encrypted communication and cybersecurity metrics for drone swarms
"""

import time
import numpy as np
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
import secrets
import json


class SecureSwarmComm:
    """Handles encrypted communication between drones using ChaCha20-Poly1305"""
    
    def __init__(self, num_drones, enable_encryption=True):
        self.num_drones = num_drones
        self.enable_encryption = enable_encryption
        
        # Generate shared key for the swarm
        if enable_encryption:
            self.key = ChaCha20Poly1305.generate_key()
            self.cipher = ChaCha20Poly1305(self.key)
        else:
            self.key = None
            self.cipher = None
        
        # Timing statistics
        self.encryption_times = []
        self.decryption_times = []
        self.total_latencies = []
    
    def encrypt_message(self, message_dict):
        """Encrypt a message dictionary"""
        if not self.enable_encryption:
            return message_dict, 0.0
        
        start_time = time.perf_counter()
        
        # Serialize message
        message_bytes = json.dumps(message_dict).encode('utf-8')
        
        # Generate nonce
        nonce = secrets.token_bytes(12)
        
        # Encrypt
        ciphertext = self.cipher.encrypt(nonce, message_bytes, None)
        
        encryption_time = (time.perf_counter() - start_time) * 1000  # ms
        self.encryption_times.append(encryption_time)
        
        encrypted_message = {
            'nonce': nonce.hex(),
            'ciphertext': ciphertext.hex(),
            'encrypted': True
        }
        
        return encrypted_message, encryption_time
    
    def decrypt_message(self, encrypted_message):
        """Decrypt a message"""
        if not self.enable_encryption:
            return encrypted_message, 0.0
        
        start_time = time.perf_counter()
        
        # Extract nonce and ciphertext
        nonce = bytes.fromhex(encrypted_message['nonce'])
        ciphertext = bytes.fromhex(encrypted_message['ciphertext'])
        
        # Decrypt
        message_bytes = self.cipher.decrypt(nonce, ciphertext, None)
        
        # Deserialize
        message_dict = json.loads(message_bytes.decode('utf-8'))
        
        decryption_time = (time.perf_counter() - start_time) * 1000  # ms
        self.decryption_times.append(decryption_time)
        
        return message_dict, decryption_time
    
    def broadcast_message(self, sender_id, message_dict, drone_positions, comm_radius):
        """
        Broadcast a message to all drones within communication radius
        Returns: list of (receiver_id, encrypted_message, latency_ms)
        """
        sender_pos = drone_positions[sender_id]
        neighbors = []
        
        for i in range(self.num_drones):
            if i == sender_id:
                continue
            
            receiver_pos = drone_positions[i]
            distance = np.linalg.norm(sender_pos - receiver_pos)
            
            if distance <= comm_radius:
                # Encrypt message
                encrypted_msg, enc_time = self.encrypt_message(message_dict)
                
                # Simulate network latency based on distance
                propagation_delay = (distance / 300.0) * 1000  # Speed of light approximation (ms)
                total_latency = enc_time + propagation_delay
                
                self.total_latencies.append(total_latency)
                neighbors.append((i, encrypted_msg, total_latency))
        
        return neighbors
    
    def receive_messages(self, messages):
        """
        Decrypt received messages
        Returns: list of (decrypted_message, total_latency)
        """
        decrypted_messages = []
        
        for encrypted_msg, recv_latency in messages:
            decrypted_msg, dec_time = self.decrypt_message(encrypted_msg)
            total_latency = recv_latency + dec_time
            decrypted_messages.append((decrypted_msg, total_latency))
        
        return decrypted_messages
    
    def get_timing_stats(self):
        """Get communication timing statistics"""
        if not self.encryption_times:
            return {
                'mean_encryption_ms': 0.0,
                'max_encryption_ms': 0.0,
                'mean_decryption_ms': 0.0,
                'max_decryption_ms': 0.0,
                'mean_total_latency_ms': 0.0,
                'max_total_latency_ms': 0.0
            }
        
        return {
            'mean_encryption_ms': np.mean(self.encryption_times),
            'max_encryption_ms': np.max(self.encryption_times),
            'mean_decryption_ms': np.mean(self.decryption_times),
            'max_decryption_ms': np.max(self.decryption_times),
            'mean_total_latency_ms': np.mean(self.total_latencies),
            'max_total_latency_ms': np.max(self.total_latencies)
        }


class CyberMetrics:
    """Tracks cybersecurity and performance metrics for the swarm"""
    
    def __init__(self, num_drones):
        self.num_drones = num_drones
        
        # Metrics storage
        self.crypto_operations = [0] * num_drones
        self.crypto_energy_j = [0.0] * num_drones
        self.communication_latencies = []
        self.control_loop_delays = []
        self.formation_errors = []
        
        # Timing
        self.last_report_time = 0.0
        self.report_interval = 5.0  # seconds
    
    def log_crypto_operation(self, drone_id, duration_s):
        """Log a cryptographic operation"""
        self.crypto_operations[drone_id] += 1
        # Assume 7.5 mW power draw during crypto operation
        self.crypto_energy_j[drone_id] += 0.0075 * duration_s
    
    def log_communication_latency(self, encryption_time_ms, decryption_time_ms, total_latency_ms):
        """Log communication latency"""
        self.communication_latencies.append({
            'encryption': encryption_time_ms,
            'decryption': decryption_time_ms,
            'total': total_latency_ms
        })
    
    def log_control_loop_delay(self, delay_ms):
        """Log control loop processing delay"""
        self.control_loop_delays.append(delay_ms)
    
    def log_formation_error(self, drone_positions, target_positions):
        """Log formation tracking error"""
        errors = np.linalg.norm(drone_positions - target_positions, axis=1)
        mean_error = np.mean(errors)
        max_error = np.max(errors)
        self.formation_errors.append({
            'mean': mean_error,
            'max': max_error
        })
    
    def print_periodic_report(self, current_time=None, force=False):
        """Print periodic metrics report"""
        if current_time is None:
            return
        
        if not force and (current_time - self.last_report_time) < self.report_interval:
            return
        
        self.last_report_time = current_time
        
        print("\n" + "="*70)
        print(f"CYBERSECURITY METRICS REPORT (t={current_time:.1f}s)")
        print("="*70)
        
        # Crypto operations
        total_ops = sum(self.crypto_operations)
        print(f"Cryptographic Operations: {total_ops} total")
        for i in range(self.num_drones):
            print(f"  Drone {i}: {self.crypto_operations[i]} ops, "
                  f"{self.crypto_energy_j[i]*1000:.2f} mJ")
        
        # Communication latency
        if self.communication_latencies:
            recent_latencies = self.communication_latencies[-100:]
            avg_latency = np.mean([l['total'] for l in recent_latencies])
            print(f"\nAverage Communication Latency: {avg_latency:.3f} ms")
        
        # Control loop delay
        if self.control_loop_delays:
            recent_delays = self.control_loop_delays[-100:]
            avg_delay = np.mean(recent_delays)
            print(f"Average Control Loop Delay: {avg_delay:.3f} ms")
        
        # Formation error
        if self.formation_errors:
            recent_errors = self.formation_errors[-100:]
            avg_error = np.mean([e['mean'] for e in recent_errors])
            print(f"Average Formation Error: {avg_error:.3f} m")
        
        print("="*70 + "\n")
    
    def export_to_dict(self):
        """Export all metrics to a dictionary"""
        return {
            'total_crypto_operations': self.crypto_operations,
            'crypto_energy_mj': [e * 1000 for e in self.crypto_energy_j],
            'communication_latencies': self.communication_latencies,
            'control_loop_delays': self.control_loop_delays,
            'formation_errors': self.formation_errors
        }
