package rittenhouse.XWING_DEV;

import java.text.DecimalFormat;
import java.util.ArrayDeque;

import rittenhouse.SPT.*;
import rittenhouse.XWINGS.*;

public class XWingFactory {
	
	private String name;
	private Location factoryLoc;
	private ArrayDeque<XWingFighter> XWingWarehouse;
	private int buildCapacity;
	
	public XWingFactory(String name, Location factoryLoc, int buildCapacity) {
		this.name = name;
		this.factoryLoc = factoryLoc;
		this.buildCapacity = buildCapacity;
		
		this.XWingWarehouse = new ArrayDeque<>();
	}

	public String getName() {
		return name;
	}

	public void setName(String name) {
		this.name = name;
	}

	public Location getFactoryLoc() {
		return factoryLoc;
	}

	public void setFactoryLoc(Location factoryLoc) {
		this.factoryLoc = factoryLoc;
	}

	public ArrayDeque<XWingFighter> getXWingWarehouse() {
		return XWingWarehouse;
	}

	public void setXWingWarehouse(ArrayDeque<XWingFighter> xWingWarehouse) {
		XWingWarehouse = xWingWarehouse;
	}

	public int getBuildCapacity() {
		return buildCapacity;
	}

	public void setBuildCapacity(int buildCapacity) {
		this.buildCapacity = buildCapacity;
	}
	
	public boolean buildFighters(int T65Count, int T70Count, int T85Count) {
		
		int totalCount = T65Count + T70Count + T85Count;
		double costT65 = 0; double costT70 = 0; double costT85 = 0;
		
		if(this.XWingWarehouse.size() + totalCount <= this.buildCapacity) {
			for(int x = 1; x <= T65Count; x ++) {
				XWingT65 y = new XWingT65("T65-" + x);
				this.XWingWarehouse.add(y);
				costT65 += y.costToBuild();
			}
			for(int x = 1; x <= T70Count; x ++) {
				XWingT70 y = new XWingT70("T70-" + x);
				this.XWingWarehouse.add(y);
				costT70 += y.costToBuild();
			}
			for(int x = 1; x <= T85Count; x ++) {
				XWingT85 y = new XWingT85("T85-" + x);
				this.XWingWarehouse.add(y);
				costT85 += y.costToBuild();
			}
			
			DecimalFormat df = new DecimalFormat("###,###.##");
			
			System.out.println("______XWing Fighter Build Report______");
			System.out.println("Type     Number Built   Build Cost");
			System.out.printf("%3s%13s%8s%-24s\n", "T65", T65Count, "", df.format(costT65) + " credits");
			System.out.printf("%3s%13s%8s%-24s\n", "T70", T70Count, "", df.format(costT70) + " credits");
			System.out.printf("%3s%13s%8s%-24s\n", "T85", T85Count, "", df.format(costT85) + " credits");
			System.out.println("______Report Created By " + this.name + " Factory");
			
			return true;
		}
		else {
			System.out.println("Fighter Build Failed");
			
			return false;
		}
	}

}
