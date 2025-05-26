import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import dblquad

class MobiusStrip:
    def __init__(self, R=1.0, w=0.5, n=100):
        """
        Initialize a Mobius strip with given parameters.
        
        Args:
            R (float): Radius of the strip (distance from center)
            w (float): Width of the strip
            n (int): Number of points in the mesh grid
        """
        self.R = R
        self.w = w
        self.n = n
        self.u = np.linspace(0, 2*np.pi, n)
        self.v = np.linspace(-w/2, w/2, n)
        self.U, self.V = np.meshgrid(self.u, self.v)
        
    def parametric_equations(self, u, v):
        """
        Compute the parametric equations for the Mobius strip.
        
        Args:
            u (float): Parameter u ∈ [0, 2π]
            v (float): Parameter v ∈ [-w/2, w/2]
            
        Returns:
            tuple: (x, y, z) coordinates
        """
        x = (self.R + v * np.cos(u/2)) * np.cos(u)
        y = (self.R + v * np.cos(u/2)) * np.sin(u)
        z = v * np.sin(u/2)
        return x, y, z
    
    def compute_mesh(self):
        """
        Compute the 3D mesh points of the Mobius strip.
        
        Returns:
            tuple: (X, Y, Z) mesh coordinates
        """
        X = (self.R + self.V * np.cos(self.U/2)) * np.cos(self.U)
        Y = (self.R + self.V * np.cos(self.U/2)) * np.sin(self.U)
        Z = self.V * np.sin(self.U/2)
        return X, Y, Z
    
    def surface_area_element(self, u, v):
        """
        Compute the surface area element for numerical integration.
        
        Args:
            u (float): Parameter u
            v (float): Parameter v
            
        Returns:
            float: Surface area element
        """
        # Partial derivatives
        dx_du = -(self.R + v * np.cos(u/2)) * np.sin(u) - (v/2) * np.sin(u/2) * np.cos(u)
        dy_du = (self.R + v * np.cos(u/2)) * np.cos(u) - (v/2) * np.sin(u/2) * np.sin(u)
        dz_du = (v/2) * np.cos(u/2)
        
        dx_dv = np.cos(u/2) * np.cos(u)
        dy_dv = np.cos(u/2) * np.sin(u)
        dz_dv = np.sin(u/2)
        
        # Cross product magnitude
        E = dx_du**2 + dy_du**2 + dz_du**2
        F = dx_du*dx_dv + dy_du*dy_dv + dz_du*dz_dv
        G = dx_dv**2 + dy_dv**2 + dz_dv**2
        
        return np.sqrt(E*G - F**2)
    
    def compute_surface_area(self):
        """
        Compute the surface area using numerical integration.
        
        Returns:
            float: Surface area
        """
        area, _ = dblquad(
            lambda u, v: self.surface_area_element(u, v),
            0, 2*np.pi,
            lambda u: -self.w/2,
            lambda u: self.w/2
        )
        return area
    
    def compute_edge_length(self):
        """
        Compute the edge length of the Mobius strip.
        
        Returns:
            float: Edge length
        """
        # Edge length is the perimeter of the strip
        # For a Mobius strip, it's 2πR
        return 2 * np.pi * self.R
    
    def plot(self):
        """
        Create a 3D visualization of the Mobius strip.
        """
        X, Y, Z = self.compute_mesh()
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot the surface
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
        
        # Add colorbar
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        
        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Mobius Strip')
        
        plt.show()

def main():
    # Create a Mobius strip instance
    mobius = MobiusStrip(R=2.0, w=1.0, n=100)
    
    # Compute and print properties
    surface_area = mobius.compute_surface_area()
    edge_length = mobius.compute_edge_length()
    
    print(f"Surface Area: {surface_area:.4f}")
    print(f"Edge Length: {edge_length:.4f}")
    
    # Visualize the strip
    mobius.plot()

if __name__ == "__main__":
    main() 