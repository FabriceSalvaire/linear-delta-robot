H = 1000;
R = 400;
r = 70;
cw = 40;
ch = cw;
z = 500;
N = [0, 0, 100]; 
for (i = [0:2])
{
	rotate(a=i*120)
	{
		translate([R, 0, 0])
		{	
 	 		cylinder(h=H, d=10, $fn=100);
 	 		translate([20, 0 , 0])
 	 		{
  		  		cylinder(h=H, d=10, $fn=100);
			};
			translate([-30,-cw/2,z])
			{	
				color ("red") cube([100,cw,ch]);
			};
		};
	};
};
translate(N)
{
	color ("blue")
	{
		cylinder(h=5, r=r, $fn=100);
	};
}