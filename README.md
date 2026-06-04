Point Cloud Scans to BIM/IFC4
Google Colab Project that takes a 3D point cloud of a building (a .ply from a LIDAR scan) and creates a IFC4 BIM of the room. The trick is that it doesn't try to do the segmentation in 3D, it flattens the scan into a top-down floor plan image, segments that with SAM 3, and then projects the result back into 3D. 

Jackson Matsumura, Ian Mendoza, Finn Wood, Marvin Recio
