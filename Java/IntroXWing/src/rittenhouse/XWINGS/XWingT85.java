package rittenhouse.XWINGS;

import rittenhouse.ENUMS.*;
import rittenhouse.SPT.Location;

public class XWingT85 extends XWingFighter {
	
	private static final long serialVersionUID = 1L;
	private Location pos;

	public XWingT85(String fighterID) {
		super(fighterID, XModel.T85, 15.68, 13.65, 2.7, 15.68, 30.1);
		this.setFSensors(XSensor.ANr1v3);
		this.setFEngines(XEngine.FTE5L9);
		this.setFPrimaryWpns(XWeapon.KX14);
		this.setFSecondaryWpns(XWeapon.MG7A);
		this.pos = new Location(10,10);
		
	}

	public Location getPos() {
		return pos;
	}

	public void setPos(Location pos) {
		this.pos = pos;
	}
	
	public void displayT85Location() {
		System.out.println("coords-> [X: " + this.pos.getX() + ", Y: " + this.pos.getY() + "]");
	}

	@Override
	public void displayFighterSpecs() {
		System.out.println("++++++T85++++++");
		System.out.println("MODEL: " + this.getFighterModel().name());
		System.out.println("Fighter ID: " + this.getFighterID());
		System.out.println("Length: " + this.getFLength());
		System.out.println("Width: " + this.getFWidth());
		System.out.println("Height: " + this.getFHeight());
		System.out.println("Mass: " + this.getFMass());
		System.out.println("Engines: " + this.getFEngines().name());
		System.out.println("Sensors: " + this.getFSensors().name());
		System.out.println("Laser Cannons: " + this.getFPrimaryWpns().name());
		System.out.println("Torpedo Launcher: " + this.getFSecondaryWpns().name());
		System.out.println("+++++++++++++++");
		
	}

	@Override
	public double costToBuild() {
		double totalCost = 0;
		
		double mc = this.getFighterModel().getModelCost() * 0.85;
		int ec = this.getFEngines().getEngineCost();
		int sc = this.getFSensors().getSensorCost();
		int pwc = this.getFPrimaryWpns().getWeaponCost();
		int swc = this.getFSecondaryWpns().getWeaponCost();
		
		totalCost = mc + ec + sc + pwc + swc;
		
		return totalCost;
	}

}
