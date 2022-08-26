package rittenhouse.XWINGS;

import rittenhouse.ENUMS.*;
import rittenhouse.SPT.Location;

public class XWingT70 extends XWingFighter {
	
	private static final long serialVersionUID = 1L;
	private Location pos;
	private XWeapon FTertiaryWpns;

	public XWingT70(String fighterID) {
		super(fighterID, XModel.T70, 12.49, 11.6, 1.92, 12.49, 29.2);
		this.setFSensors(XSensor.ANq5v8);
		this.setFEngines(XEngine.FTE5L5);
		this.setFPrimaryWpns(XWeapon.KX12);
		this.setFSecondaryWpns(XWeapon.MG7A);
		this.setFTertiaryWpns(XWeapon.AX190);
		this.pos = new Location(10,10);
	}

	public Location getPos() {
		return pos;
	}

	public void setPos(Location pos) {
		this.pos = pos;
	}

	public XWeapon getFTertiaryWpns() {
		return FTertiaryWpns;
	}

	public void setFTertiaryWpns(XWeapon fTertiaryWpns) {
		FTertiaryWpns = fTertiaryWpns;
	}
	
	public void displayT70Location() {
		System.out.println("coords-> [X: " + this.pos.getX() + ", Y: " + this.pos.getY() + "]");
	}

	@Override
	public void displayFighterSpecs() {
		System.out.println("++++++T70++++++");
		System.out.println("MODEL: " + this.getFighterModel().name());
		System.out.println("Fighter ID: " + this.getFighterID());
		System.out.println("Length: " + this.getFLength());
		System.out.println("Width: " + this.getFWidth());
		System.out.println("Height: " + this.getFHeight());
		System.out.println("Mass: " + this.getFMass());
		System.out.println("Engines: " + this.getFEngines().name());
		System.out.println("Sensors: " + this.getFSensors().name());
		System.out.println("Laser Cannons: " + this.getFPrimaryWpns().name());
		System.out.println("Heavy Laser Cannons: " + this.getFTertiaryWpns().name());
		System.out.println("Torpedo Launcher: " + this.getFSecondaryWpns().name());
		System.out.println("+++++++++++++++");
		
	}

	@Override
	public double costToBuild() {
		double totalCost = 0;
		
		double mc = this.getFighterModel().getModelCost() * 1.5;
		int ec = this.getFEngines().getEngineCost();
		int sc = this.getFSensors().getSensorCost();
		int pwc = this.getFPrimaryWpns().getWeaponCost();
		int swc = this.getFSecondaryWpns().getWeaponCost();
		int twc = this.getFTertiaryWpns().getWeaponCost();
		
		totalCost = mc + ec + sc + pwc + swc + twc;
		
		return totalCost;
	}

}
