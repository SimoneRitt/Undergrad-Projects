package rittenhouse.PARTICLE;

import java.text.DecimalFormat;

public class Particle {
	
	// variables
	
	private String retrievalCode;
	private double pDiameter;
	private double pVolume;
	private double pSurfaceArea;
	private String pComposition;
	private boolean isRefined;
	
	public static int particleCount = 0;
	
	// constructor
	
	public Particle(double pDiameter, String pComposition) {
		
		this.pDiameter = pDiameter;
		this.pComposition = pComposition;
		this.retrievalCode = "P" + particleCount;
		this.isRefined = false;
		this.pVolume = (4.0/3.0) * Math.PI * Math.pow((pDiameter/2), 3);
		this.pSurfaceArea = 4.0 * Math.PI * Math.pow((pDiameter/2), 2);
		
		particleCount += 1;
	}
	
	// getters and setters
	
	public String getRetrievalCode() {
		return retrievalCode;
	}

	public void setRetrievalCode(String retrievalCode) {
		this.retrievalCode = retrievalCode;
	}

	public double getpDiameter() {
		return pDiameter;
	}

	public void setpDiameter(double pDiameter) {
		this.pDiameter = pDiameter;
	}

	public double getpVolume() {
		return pVolume;
	}

	public void setpVolume(double pVolume) {
		this.pVolume = pVolume;
	}

	public double getpSurfaceArea() {
		return pSurfaceArea;
	}

	public void setpSurfaceArea(double pSurfaceArea) {
		this.pSurfaceArea = pSurfaceArea;
	}

	public String getpComposition() {
		return pComposition;
	}

	public void setpComposition(String pComposition) {
		this.pComposition = pComposition;
	}

	public boolean isRefined() {
		return isRefined;
	}

	public void setRefined(boolean isRefined) {
		this.isRefined = isRefined;
	}
	
	// methods

	public void displayParticleInfo() {
		System.out.println("Particle " + this.retrievalCode);
		
		DecimalFormat df1 = new DecimalFormat("##");
		
		System.out.println("Diameter: " + df1.format(this.pDiameter) + " microns");
		
		DecimalFormat df2 = new DecimalFormat("##.##");
		
		System.out.println("Volume: " + df2.format(this.pVolume) + " cubic microns");
		System.out.println("Surface Area: " + df2.format(this.pSurfaceArea) + " squared microns");
		System.out.println("Composition: " + this.pComposition);
		
		if(this.isRefined) {
			System.out.println("Refined:  has been Refined");
		}
		else {
			System.out.println("Refined:  has not been Refined");
		}
		
		System.out.println("********************");
		
	}

}
