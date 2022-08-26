package rittenhouse.XWINGS;

import rittenhouse.ENUMS.*;
import rittenhouse.SPT.Location;

public class XWingT65 extends XWingFighter {
	
	private static final long serialVersionUID = 1L;
	private Location pos;

	public XWingT65(String fighterID) {
		super(fighterID, XModel.T65B, 13.4, 11.6, 2.4, 13.4, 26.2);
		this.setFSensors(XSensor.ANs5D);
		this.setFEngines(XEngine.CPG04Z);
		this.setFPrimaryWpns(XWeapon.KX9);
		this.setFSecondaryWpns(XWeapon.MG7);
		this.pos = new Location(10,10);
		
	}

	public Location getPos() {
		return pos;
	}

	public void setPos(Location pos) {
		this.pos = pos;
	}

	public void displayT65Location() {
		System.out.println("coords-> [X: " + this.pos.getX() + ", Y: " + this.pos.getY() + "]");
	}
	
	@Override
	public void displayFighterSpecs() {
		System.out.println("++++++T65B++++++");
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
		System.out.println("++++++++++++++++");
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
