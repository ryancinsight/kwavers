use super::*;

#[test]
fn test_soa_storage_creation() {
    let mut storage = SoAFieldStorage::new(4, 1000).expect("SoA must create");

    assert_eq!(storage.num_fields(), 4);
    assert_eq!(storage.num_elements(), 1000);
    assert!(storage.total_bytes() >= 4 * 1000 * 8);

    let field0 = storage.field_mut(0).expect("field 0 must exist");
    assert_eq!(field0.len(), 1000);

    field0[0] = 1.0;
    field0[999] = 2.0;

    let field0_read = storage.field(0).expect("field 0 must read");
    assert_eq!(field0_read[0], 1.0);
    assert_eq!(field0_read[999], 2.0);
}

#[test]
fn test_soa_to_aos_conversion() {
    let mut soa = SoAFieldStorage::new(3, 10).expect("SoA must create");

    {
        let f0 = soa.field_mut(0).unwrap();
        for (i, v) in f0.iter_mut().enumerate().take(10) {
            *v = i as f64 * 1.0;
        }
    }
    {
        let f1 = soa.field_mut(1).unwrap();
        for (i, v) in f1.iter_mut().enumerate().take(10) {
            *v = i as f64 * 10.0;
        }
    }
    {
        let f2 = soa.field_mut(2).unwrap();
        for (i, v) in f2.iter_mut().enumerate().take(10) {
            *v = i as f64 * 100.0;
        }
    }

    let mut aos = vec![0.0; 30];
    soa.to_aos(&mut aos).expect("conversion must succeed");

    // AoS layout: [f0[0], f1[0], f2[0], f0[1], f1[1], f2[1], ...]
    assert_eq!(aos[0], 0.0); // f0[0] = 0*1
    assert_eq!(aos[1], 0.0); // f1[0] = 0*10
    assert_eq!(aos[2], 0.0); // f2[0] = 0*100
    assert_eq!(aos[3], 1.0); // f0[1] = 1*1
    assert_eq!(aos[4], 10.0); // f1[1] = 1*10
    assert_eq!(aos[5], 100.0); // f2[1] = 1*100
}

#[test]
fn test_copy_from_slices() {
    let mut soa = SoAFieldStorage::new(3, 5).unwrap();

    let src0: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let src1: Vec<f64> = vec![10.0, 20.0, 30.0, 40.0, 50.0];
    let src2: Vec<f64> = vec![100.0, 200.0, 300.0, 400.0, 500.0];

    let slices: Vec<&[f64]> = vec![&src0, &src1, &src2];
    soa.copy_from_slices(&slices).expect("copy must succeed");

    assert_eq!(soa.field(0).unwrap(), &[1.0, 2.0, 3.0, 4.0, 5.0]);
    assert_eq!(soa.field(1).unwrap(), &[10.0, 20.0, 30.0, 40.0, 50.0]);
    assert_eq!(soa.field(2).unwrap(), &[100.0, 200.0, 300.0, 400.0, 500.0]);
}
