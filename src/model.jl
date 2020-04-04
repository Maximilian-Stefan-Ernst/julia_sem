mutable struct model{
        RAM,
        PAR,
        DATA,
        MSTRUC,
        LOGL,
        OPT,
        EST,
        OBS_COV,
        IMP_COV,
        OBS_MEAN,
        OPT_RESULT,
        SE,
        Z,
        P,
        REG}
    ram::RAM
    par::PAR
    data::DATA
    mstruc::MSTRUC
    logl::LOGL
    opt::OPT
    est::EST
    obs_cov::OBS_COV
    imp_cov::IMP_COV
    obs_mean::OBS_MEAN
    opt_result::OPT_RESULT
    se::SE
    z::Z
    p::P
    reg::REG
    model{RAM, PAR, DATA, MSTRUC, LOGL, OPT, EST, OBS_COV, IMP_COV, OBS_MEAN, OPT_RESULT, SE, Z, P, REG}(
            ram, par, data,
            mstruc,
            logl,
            opt,
            est,
            obs_cov,
            imp_cov,
            obs_mean,
            opt_result,
            se,
            z,
            p,
            reg) where {
            RAM,
            PAR,
            DATA,
            MSTRUC,
            LOGL,
            OPT,
            EST,
            OBS_COV,
            IMP_COV,
            OBS_MEAN,
            OPT_RESULT,
            SE,
            Z,
            P,
            REG} =
    new(ram, par, data,
            mstruc,
            logl,
            opt,
            est,
            obs_cov,
            imp_cov,
            obs_mean,
            opt_result,
            se,
            z,
            p,
            reg)
end


# second outer constructor for @set
model(ram::RAM, par::PAR, data::DATA = nothing,
        mstruc::MSTRUC = false,
        logl::LOGL = nothing,
        opt::OPT = "LBFGS",
        est::EST = nothing,
        obs_cov::OBS_COV = nothing,
        imp_cov::IMP_COV = nothing,
        obs_mean::OBS_MEAN = nothing,
        opt_result::OPT_RESULT = nothing,
        se::SE = nothing,
        z::Z = nothing,
        p::P = nothing,
        reg::REG = nothing) where {
        RAM,
        DATA,
        PAR,
        MSTRUC,
        LOGL,
        OPT,
        EST,
        OBS_COV,
        IMP_COV,
        OBS_MEAN,
        OPT_RESULT,
        SE,
        Z,
        P,
        REG} =
        model{RAM, PAR, DATA, MSTRUC, LOGL, OPT, EST, OBS_COV, IMP_COV, OBS_MEAN, OPT_RESULT, SE, Z, P, REG}(
                ram, par, data,
                mstruc,
                logl,
                opt,
                est,
                obs_cov,
                imp_cov,
                obs_mean,
                opt_result,
                se,
                z,
                p,
                reg)
### untyped struct for user

mutable struct sem_model
        ram
        par
        data
        mstruc
        logl
        opt
        est
        obs_cov
        imp_cov
        obs_mean
        opt_result
        se
        z
        p
        reg
end

sem_model(ram, par;
        data = nothing,
        mstruc = false,
        logl = nothing,
        opt = "LBFGS",
        est = nothing,
        obs_cov = nothing,
        imp_cov = nothing,
        obs_mean = nothing,
        opt_result = nothing,
        se = nothing,
        z = nothing,
        p = nothing,
        reg = nothing) =
        sem_model(ram, par,
                data,
                mstruc,
                logl,
                opt,
                est,
                obs_cov,
                imp_cov,
                obs_mean,
                opt_result,
                se,
                z,
                p,
                reg)



model(sem::sem_model) = model(sem.ram, sem.par,
                        sem.data,
                        sem.mstruc,
                        sem.logl,
                        sem.opt,
                        sem.est,
                        sem.obs_cov,
                        sem.imp_cov,
                        sem.obs_mean,
                        sem.opt_result,
                        sem.se,
                        sem.z,
                        sem.p,
                        sem.reg)


mutable struct reg{LASSO, LASSO_PEN, RIDGE, RIDGE_PEN}
        lasso::LASSO
        lasso_pen::LASSO_PEN
        ridge::RIDGE
        ridge_pen::RIDGE_PEN
        reg{LASSO, LASSO_PEN, RIDGE, RIDGE_PEN}(
                lasso, lasso_pen, ridge, ridge_pen) where {
                LASSO, LASSO_PEN, RIDGE, RIDGE_PEN} =
        new(lasso, lasso_pen, ridge, ridge_pen)
end


reg(lasso::LASSO = nothing, lasso_pen::LASSO_PEN = nothing,
        ridge::RIDGE = nothing, ridge_pen::RIDGE_PEN = nothing) where {
        LASSO, LASSO_PEN, RIDGE, RIDGE_PEN} =
        reg{LASSO, LASSO_PEN, RIDGE, RIDGE_PEN}(
                lasso, lasso_pen, ridge, ridge_pen
                )
